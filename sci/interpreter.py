import torch
from torch import nn
from .sp import compute_sp


class Interpreter(nn.Module):
    """
    A lightweight SCI interpreter.

    - Encodes input features into a hidden representation
    - Emits marker logits (for SP)
    - Emits task logits (for classification)

    This is a minimal prototype; in practice you would replace
    the feature encoder with a CNN/Transformer/etc.
    """

    def __init__(self, feature_dim: int = 128, num_markers: int = 8, num_classes: int = 10):
        super().__init__()
        self.encoder = nn.Linear(feature_dim, feature_dim)
        self.marker_head = nn.Linear(feature_dim, num_markers)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.encoder(x))
        return h

    def compute(self, x: torch.Tensor):
        """
        Compute SP and predictions for a batch of inputs.

        Args:
            x: tensor of shape (batch_size, feature_dim)

        Returns:
            sp_mean: scalar SP value (mean over batch)
            logits: tensor of shape (batch_size, num_classes)
        """
        h = self.encode(x)
        marker_logits = self.marker_head(h)
        sp = compute_sp(marker_logits)  # (batch_size,)
        logits = self.classifier(h)
        return sp.mean(), logits

    def grad_sp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient of SP wrt parameters as a flat vector.
        NOTE: This assumes gradients have been zeroed before calling.
        """
        self.zero_grad()
        sp, _ = self.compute(x)
        sp.backward()
        grads = []
        for p in self.parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1))
        if not grads:
            return torch.zeros(0)
        return torch.cat(grads).detach()

    @torch.no_grad()
    def parameters_vector(self) -> torch.Tensor:
        """Flatten all parameters into a single vector."""
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    @torch.no_grad()
    def update_parameters(self, new_theta: torch.Tensor) -> None:
        """Load a flat parameter vector back into the module parameters."""
        offset = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(new_theta[offset : offset + n].view_as(p))
            offset += n
