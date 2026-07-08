import torch

from .functional import forward_with_params
from .losses import vae_loss


class InnerLoop:
    """Vanilla (second-order) MAML inner loop."""

    def __init__(self, model, inner_lr, inner_steps=1, beta=1.0):
        self.model = model
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.beta = beta

    def adapt(self, support_lr, support_hr):
        params = {name: p for name, p in self.model.named_parameters()}

        for _ in range(self.inner_steps):
            reconstruction, mu, log_var = forward_with_params(
                self.model, params, (support_lr,)
            )
            loss, _, _ = vae_loss(reconstruction, support_hr, mu, log_var, self.beta)

            grads = torch.autograd.grad(
                loss, params.values(), create_graph=True
            )

            params = {
                name: param - self.inner_lr * grad
                for (name, param), grad in zip(params.items(), grads)
            }

        return params
