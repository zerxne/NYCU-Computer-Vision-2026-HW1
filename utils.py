import math
import copy
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler


# ---------------------------------------------------------------------------
# SAM Optimizer
# ---------------------------------------------------------------------------

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho: float = 0.05,
                 adaptive: bool = False, **kwargs):
        assert rho >= 0.0, f"Invalid rho: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (
                    (torch.pow(p, 2) if group["adaptive"] else 1.0)
                    * p.grad
                    * scale.to(p)
                )
                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]

        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "SAM requires a closure."
        closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


# ---------------------------------------------------------------------------
# Label Smoothing Cross Entropy
# ---------------------------------------------------------------------------

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n = logits.size(-1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth = torch.full_like(log_probs, self.smoothing / (n - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        return -(smooth * log_probs).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for (name, ema_p), (_, model_p) in zip(
            self.module.named_parameters(), model.named_parameters()
        ):
            ema_p.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)
        for (name, ema_b), (_, model_b) in zip(
            self.module.named_buffers(), model.named_buffers()
        ):
            ema_b.copy_(model_b.data)


# ---------------------------------------------------------------------------
# LR Scheduler
# ---------------------------------------------------------------------------

class CosineAnnealingWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int,
                 min_lr: float = 1e-6, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        e = self.last_epoch
        if e < self.warmup_epochs:
            scale = (e + 1) / max(1, self.warmup_epochs)
            return [base * scale for base in self.base_lrs]
        t = (e - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
        cosine = 0.5 * (1 + math.cos(math.pi * t))
        return [self.min_lr + (base - self.min_lr) * cosine for base in self.base_lrs]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        bs = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        correct = pred.t().eq(target.view(1, -1).expand_as(pred.t()))
        return [correct[:k].reshape(-1).float().sum().mul_(100.0 / bs).item() for k in topk]
