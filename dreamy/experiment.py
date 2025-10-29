import os
import dataclasses
import pickle
from typing import Callable
import torch
from dreamy.epo import epo
from dreamy.epo import load_model  # make sure load_model is available here


def chunk_list(lst, N):
    avg = len(lst) // N
    rem = len(lst) % N
    start = 0

    for i in range(N):
        end = start + avg + (i < rem)
        yield lst[start:end]
        start = end


def retrieve_files(cfgs):
    for c in cfgs:
        import s3fs
        fs = s3fs.S3FileSystem()
        s3_full_path = os.path.join(c.s3_bucket, c.s3_path)
        print("downloading", s3_full_path, "to", c.output_path)
        try:
            fs.download(s3_full_path, c.output_path)
        except FileNotFoundError:
            print("file not found", s3_full_path)


def check_file_exists(s3, bucket, key):
    from botocore.exceptions import ClientError
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        else:
            raise


@dataclasses.dataclass
class DreamConfig:
    """
    Local dream config (no Modal / cluster stuff).
    """

    runner_builder: Callable
    model_size: str = "12b"

    # steering settings
    minimize: bool = True  # push activation DOWN instead of up

    # search / optimization hyperparams
    x_penalty_min: float = 1.0 / 10.0
    x_penalty_max: float = 10.0
    iters: int = 300
    seq_len: int = 12
    batch_size: int = 256
    population_size: int = 16
    explore_per_pop: int = 64

    # restart heuristics
    restart_frequency: int = 30
    restart_xentropy: float = 2.0
    restart_xentropy_max_mult: float = 3.0

    # init prompt
    initial_ids: torch.Tensor = None
    initial_str: str = ""  # overrides initial_ids

    # scoring / logging
    topk: int = 512
    attribution_frequency: int = None

    # output / persistence
    output_path: str = ""
    s3_bucket: str = "caiplay"
    s3_path: str = ""

    # misc
    seed: int = 0
    payload: dict = None

    # GCG mode shortcut
    gcg: float = None  # if set, overrides some params


def dream(c: DreamConfig, model=None, tokenizer=None):
    """
    Run one dream search locally on the current Colab GPU.

    - Uses the runner from c.runner_builder
    - Optionally loads model if not provided
    - Runs epo() with minimization support
    - Saves result locally and/or to s3 if configured
    """

    # if user is using s3 resume/skip semantics:
    if len(c.s3_path) > 0:
        import boto3
        s3 = boto3.client("s3")
        if check_file_exists(s3, c.s3_bucket, c.s3_path):
            print("run already done, skipping", c.s3_path)
            return False

    # load model if caller didn't pass one in
    if model is None or tokenizer is None:
        model, tokenizer = load_model(model_size=c.model_size)

    # GCG shortcut mode
    if c.gcg is not None:
        c.population_size = 1
        c.explore_per_pop = c.batch_size
        c.x_penalty_min = c.gcg
        c.x_penalty_max = c.gcg

    # initial prompt seeding
    if len(c.initial_str) > 0:
        c.initial_ids = tokenizer.encode(
            c.initial_str,
            return_tensors="pt"
        ).to(model.device)

    # build runner (this is where we inject your residual_runner w/ minimize=True)
    runner = c.runner_builder(model, tokenizer)
    setattr(runner, "minimize", c.minimize)

    # main optimization loop
    history = epo(
        runner,
        model,
        tokenizer,
        initial_ids=c.initial_ids,
        seed=c.seed,
        x_penalty_min=c.x_penalty_min,
        x_penalty_max=c.x_penalty_max,
        iters=c.iters,
        seq_len=c.seq_len,
        batch_size=c.batch_size,
        population_size=c.population_size,
        explore_per_pop=c.explore_per_pop,
        restart_frequency=c.restart_frequency,
        restart_xentropy=c.restart_xentropy,
        restart_xentropy_max_mult=c.restart_xentropy_max_mult,
        topk=c.topk,
    )

    # we can't pickle functions easily, so drop runner_builder before saving
    c.runner_builder = None
    output = (c, history)

    # optional: save to local disk
    if len(c.output_path) > 0:
        folder_path = os.path.dirname(c.output_path)
        os.makedirs(folder_path, exist_ok=True)
        with open(c.output_path, "wb") as f:
            pickle.dump(output, f)

    # optional: push to S3
    if len(c.s3_path) > 0:
        import boto3
        s3 = boto3.resource("s3")
        s3.Bucket(c.s3_bucket).upload_file(c.output_path, c.s3_path)

    return output
