import torch
import torch.nn.functional as F


def compute_sp(marker_logits: torch.Tensor) -> torch.Tensor:
    """
    Compute an entropy-based Surgical Precision (SP) score.

    SP = 1 - H(q) / log(K), where:
    - q = softmax(marker_logits)
    - H(q) is Shannon entropy over markers
    - K is the number of markers

    Args:
        marker_logits: tensor of shape (..., K)

    Returns:
        SP: tensor of shape (...,) with values in [0, 1].
    """
    q = F.softmax(marker_logits, dim=-1)
    k = q.shape[-1]
    entropy = -torch.sum(q * torch.log(q + 1e-9), dim=-1)
    sp = 1.0 - entropy / torch.log(torch.tensor(float(k), device=marker_logits.device))
    return sp
