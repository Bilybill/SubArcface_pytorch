from bisect import bisect_right
from .base import _WarmUpLRScheduler

__all__ = ["StepLRScheduler"]


class StepLRScheduler(_WarmUpLRScheduler):
    def __init__(
        self,
        optimizer,
        lr_steps,
        lr_mults,
        base_lr,
        warmup_lr,
        warmup_steps,
        last_iter=-1,
        max_iter=None,
    ):
        super(StepLRScheduler, self).__init__(
            optimizer, base_lr, warmup_lr, warmup_steps, last_iter
        )

        assert len(lr_steps) == len(lr_mults), "{} vs {}".format(milestone, lr_mults)
        for x in lr_steps:
            assert isinstance(x, int)
        if not list(lr_steps) == sorted(lr_steps):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                lr_steps,
            )
        self.lr_steps = lr_steps
        self.lr_mults = [1.0]
        for x in lr_mults:
            self.lr_mults.append(self.lr_mults[-1] * x)

    def _get_new_lr(self):
        warmup_lr = self._get_warmup_lr()
        if warmup_lr is not None:
            return warmup_lr

        pos = bisect_right(self.lr_steps, self.last_iter)
        scale = self.warmup_lr * self.lr_mults[pos] / self.base_lr
        return [base_lr * scale for base_lr in self.base_lrs]
