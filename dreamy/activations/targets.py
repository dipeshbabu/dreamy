from dataclasses import dataclass
import torch


@dataclass
class DirectionSpec:
    layer: int
    v: torch.Tensor  # [d_model], unit norm
