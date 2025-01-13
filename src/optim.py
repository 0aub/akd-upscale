import math
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import (
    StepLR, ExponentialLR, CosineAnnealingLR, LinearLR, LambdaLR
)

class Optimizer:
    def __init__(self, model_parameters, config, num_train_steps):
        self.model_parameters = model_parameters
        self.config = config
        self.num_train_steps = num_train_steps
        self.total_steps = self.num_train_steps * self.config.epochs
        self.warmup_steps = self._get_warmup_steps()
        
        # Initialize the underlying optimizer
        self.optimizer = self._init_optimizer()
        
        # Initialize the scheduler (if any)
        self.scheduler = self._init_scheduler()

    def _init_optimizer(self):
        optimizer_name = self.config.optimizer
        lr = self.config.learning_rate
        wd = self.config.weight_decay
        
        # Maps to classes in torch.optim
        if optimizer_name in ["Adam", "SGD", "RMSprop", "AdamW"]:
            optimizer_class = getattr(optim, optimizer_name)
            return optimizer_class(self.model_parameters, lr=lr, weight_decay=wd)
        else:
            raise ValueError("Unsupported optimizer. Available options: 'Adam', 'SGD', 'RMSprop', 'AdamW'.")

    def _get_warmup_steps(self):
        # If warmup_steps is manually > 0, use it.
        # Otherwise compute from warmup_ratio * num_train_steps
        if self.config.warmup_steps > 0:
            return self.config.warmup_steps
        return math.ceil(self.num_train_steps * self.config.warmup_ratio)

    def _warmup_constant_lr(self, current_step):
        if current_step < self.warmup_steps:
            return float(current_step) / float(max(1, self.warmup_steps))
        return 1.0

    def _init_scheduler(self):
        sched = self.config.scheduler
        if sched == "step":
            return StepLR(self.optimizer, step_size=1000, gamma=0.1)
        elif sched == "exponential":
            return ExponentialLR(self.optimizer, gamma=0.95)
        elif sched == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=500)
        elif sched == "linear":
            # linear decay from start_factor=1.0 to end_factor=0.1 
            # over self.warmup_steps steps
            return LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=self.warmup_steps)
        elif sched == "constant":
            return LambdaLR(self.optimizer, lambda _: 1.0)
        elif sched == "constant_with_warmup":
            return LambdaLR(self.optimizer, self._warmup_constant_lr)
        else:
            return None

    def get_lr(self):
        if self.scheduler:
            return self.scheduler.get_last_lr()[0]
        return self.optimizer.param_groups[0]['lr']

    def step(self):
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def clip_grad_norm(self, parameters):
        clip_grad_norm_(parameters, self.configclip_max_norm)
