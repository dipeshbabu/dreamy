from dataclasses import dataclass
import torch


@dataclass
class DirectionSpec:
    layer: int    # which gpt_neox layer index
    v: torch.Tensor    # [d_model], unit norm direction vector (same dtype/device as model)
