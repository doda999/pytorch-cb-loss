class WarmupStepScheduler(object):
    def __init__(
        self, 
        optimizer, 
        milestones,
        gamma=0.1,
        warmup_factor=1.0/3,
        warmup_epoch=5,
        warmup_method="linear",
        last_epoch=-1,
    ):
        self.optimizer = optimizer
        self.milestones = milestones
        self.factor = 1.
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_epoch = warmup_epoch
        self.warmup_method = warmup_method
        if last_epoch == -1:
            self.last_epoch = 0
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            self.last_epoch = last_epoch
        self.base_lrs = list(map(lambda group: group["initial_lr"], optimizer.param_groups))
    
    def step(self):
        self.last_epoch += 1
        if self.last_epoch <= self.warmup_epoch:
            if self.warmup_method == "constant":
                self.factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch)/self.warmup_epoch
                self.factor = self.warmup_factor * (1-alpha) + alpha
        elif self.last_epoch in self.milestones:
            self.factor *= self.gamma
        elif self.last_epoch < self.milestones[0]:
            self.factor = 1.

        new_lrs = [base_lr * self.factor for base_lr in self.base_lrs]
        for param_group, lr in zip(self.optimizer.param_groups, new_lrs):
            param_group['lr'] = lr


