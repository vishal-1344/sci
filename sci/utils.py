import torch
from torch import nn


def flatten_params(model: nn.Module) -> torch.Tensor:
    """Flatten all parameters of a model into a single vector."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])
