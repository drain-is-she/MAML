from .functional import forward_with_params
from .innerloop import InnerLoop
from .losses import vae_loss
import torch


class MetaLearner:
    def __init__(self, model, inner_lr, meta_lr, inner_steps=1, beta=1.0):
        self.model = model
        self.inner_loop = InnerLoop(model, inner_lr, inner_steps, beta)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=meta_lr)
        self.beta = beta

    def meta_train_step(self, support_lr, support_hr, query_lr, query_hr):
        adapted_params = self.inner_loop.adapt(support_lr, support_hr)

        reconstruction, mu, log_var = forward_with_params(
            self.model, adapted_params, (query_lr,)
        )

        total_loss, recon_loss, kl_loss = vae_loss(
            reconstruction, query_hr, mu, log_var, self.beta
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), recon_loss.item(), kl_loss.item()

    @torch.no_grad()
    def meta_eval_step(self, support_lr, support_hr, query_lr, query_hr):
        # adaptation itself needs grad even at eval time, so re-enable locally
        with torch.enable_grad():
            adapted_params = self.inner_loop.adapt(support_lr, support_hr)
            reconstruction, mu, log_var = forward_with_params(
                self.model, adapted_params, (query_lr,)
            )
            total_loss, recon_loss, kl_loss = vae_loss(
                reconstruction, query_hr, mu, log_var, self.beta
            )
        return total_loss.item(), recon_loss.item(), kl_loss.item(), reconstruction
