class _LRScheduler(object):
    def __init__(self, optimizer, last_iter=-1):
        self.optimizer = optimizer
        if last_iter == -1:
            for group in optimizer.param_groups:
                group["initial_lr"] = group["lr"]
            self.has_base_lrs = True
            self._get_base_lrs_later()
        else:
            self.has_base_lrs = False
        self.last_iter = last_iter

    def _get_base_lrs_later(self):
        self.base_lrs = list(
            map(lambda group: group["initial_lr"], self.optimizer.param_groups)
        )

    def _get_new_lr(self):
        raise NotImplementedError

    def get_lr(self):
        return list(map(lambda group: group["lr"], self.optimizer.param_groups))

    def step(self, this_iter=None):
        if not self.has_base_lrs:
            self._get_base_lrs_later()

        if this_iter is None:
            this_iter = self.last_iter + 1
        self.last_iter = this_iter
        for param_group, lr in zip(self.optimizer.param_groups, self._get_new_lr()):
            param_group["lr"] = lr


class _WarmUpLRScheduler(_LRScheduler):
    def __init__(self, optimizer, base_lr, warmup_lr, warmup_steps, last_iter=-1):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        if warmup_steps == 0:
            self.warmup_lr = base_lr
        else:
            self.warmup_lr = warmup_lr
        super(_WarmUpLRScheduler, self).__init__(optimizer, last_iter)

    def _get_warmup_lr(self):
        if self.warmup_steps > 0 and self.last_iter < self.warmup_steps:
            scale = (
                (self.last_iter / self.warmup_steps) * (self.warmup_lr - self.base_lr)
                + self.base_lr
            ) / self.base_lr
            return [scale * base_lr for base_lr in self.base_lrs]
        else:
            return None
