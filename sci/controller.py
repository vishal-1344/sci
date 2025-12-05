import torch
from torch import nn


class SCIController(nn.Module):
    """
    Minimal SCI closed-loop controller.

    It monitors a scalar interpretive state SP, compares it
    to a target SP*, and performs a projected gradient-style
    update on the interpreter parameters Θ based on ΔSP.

    This is a simplified, minimal prototype to show the core idea.
    """

    def __init__(
        self,
        interpreter: nn.Module,
        sp_target: float = 0.90,
        eta: float = 0.01,
        gamma: float = 0.10,
        trust_region: float = 0.1,
    ):
        super().__init__()
        self.interpreter = interpreter
        self.sp_target = sp_target
        self.eta = eta
        self.gamma = gamma
        self.trust_region = trust_region

    @torch.no_grad()
    def _project(self, theta: torch.Tensor, theta_old: torch.Tensor) -> torch.Tensor:
        """Simple trust-region projection on parameter vector."""
        delta = theta - theta_old
        norm = delta.norm()
        if norm > self.trust_region:
            return theta_old + self.trust_region * delta / (norm + 1e-9)
        return theta

    def forward(self, x: torch.Tensor):
        """
        Run a single SCI control step.

        Args:
            x: input features (batch_size, feature_dim)

        Returns:
            pred: raw predictions (logits)
            sp: scalar SP estimate (float tensor)
            d_sp: SP* - SP
            interpreter: the (possibly) updated interpreter module
        """
        sp, pred = self.interpreter.compute(x)
        d_sp = self.sp_target - sp

        # No-op zone: if |ΔSP| is small, do not update
        if torch.abs(d_sp) < self.gamma:
            return pred, sp, d_sp, self.interpreter

        # Collect old parameters as a flat vector
        theta_old = self.interpreter.parameters_vector().detach()

        # Compute gradient of SP wrt parameters
        grad = self.interpreter.grad_sp(x)

        # Basic controller update: Θ_new = Θ_old + η * ΔSP * ∇Θ SP
        theta_new = theta_old + self.eta * d_sp * grad

        # Trust-region projection
        theta_new = self._project(theta_new, theta_old)

        # Push updated parameters back into the interpreter
        self.interpreter.update_parameters(theta_new)

        return pred, sp, d_sp, self.interpreter
