import math
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import (
    StepLR, ExponentialLR, CosineAnnealingLR, LinearLR, LambdaLR
)

class Optimizer:
    def __init__(self, model_parameters, optimizer, learning_rate, weight_decay, betas, scheduler, 
                 warmup_steps, warmup_ratio, num_train_steps, epochs, clip_max_norm):
        self.model_parameters = model_parameters
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.scheduler_name = scheduler
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio
        self.num_train_steps = num_train_steps
        self.epochs = epochs
        self.clip_max_norm = clip_max_norm
        
        self.total_steps = self.num_train_steps * self.epochs
        self.warmup_steps = self._get_warmup_steps()
        
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()

    def _init_optimizer(self):
        if self.optimizer_name in ["Adam", "AdamW"]:
            optimizer_class = getattr(optim, self.optimizer_name)
            if self.betas:
                return optimizer_class(self.model_parameters, lr=self.learning_rate, weight_decay=self.weight_decay, betas=self.betas)
            return optimizer_class(self.model_parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name in ["SGD", "RMSprop"]:
            optimizer_class = getattr(optim, self.optimizer_name)
            return optimizer_class(self.model_parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError("Unsupported optimizer. Available options: 'Adam', 'SGD', 'RMSprop', 'AdamW'.")

    def _get_warmup_steps(self):
        if self.warmup_steps > 0:
            return self.warmup_steps
        return math.ceil(self.num_train_steps * self.warmup_ratio)

    def _warmup_constant_lr(self, current_step):
        if current_step < self.warmup_steps:
            return float(current_step) / float(max(1, self.warmup_steps))
        return 1.0

    def _init_scheduler(self):
        if self.scheduler_name == "step":
            return StepLR(self.optimizer, step_size=1000, gamma=0.1)
        elif self.scheduler_name == "exponential":
            return ExponentialLR(self.optimizer, gamma=0.95)
        elif self.scheduler_name == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=500)
        elif self.scheduler_name == "linear":
            return LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=self.warmup_steps)
        elif self.scheduler_name == "constant":
            return LambdaLR(self.optimizer, lambda _: 1.0)
        elif self.scheduler_name == "constant_with_warmup":
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
        return clip_grad_norm_(parameters, self.clip_max_norm)

    def state_dict(self):
        """Returns the state of the optimizer and scheduler."""
        return {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None
        }

    def load_state_dict(self, state_dict):
        """Loads the state of the optimizer and scheduler."""
        self.optimizer.load_state_dict(state_dict['optimizer'])
        if self.scheduler and state_dict['scheduler']:
            self.scheduler.load_state_dict(state_dict['scheduler'])

