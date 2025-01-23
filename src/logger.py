import time
import math
import os

def sci(num):
    if num == 0:
        return '0.0'
    return '{:.0e}'.format(num) 

class Metrics:
    def __init__(self):
        self.reset()
        self.interval_data = {}

    def reset(self):
        self.data = {}

    def reset_interval(self):
        self.interval_data = {}

    def update(self, key, value, count=1):
        for storage in [self.data, self.interval_data]:
            if key not in storage:
                storage[key] = {'sum': 0, 'count': 0}
            storage[key]['sum'] += value * count
            storage[key]['count'] += count

    def average(self, key, interval=False):
        storage = self.interval_data if interval else self.data
        if key in storage and storage[key]['count'] > 0:
            return storage[key]['sum'] / storage[key]['count']
        else:
            return 0

    def summary(self, interval=False):
        storage = self.interval_data if interval else self.data
        return {k: self.average(k, interval=interval) for k in storage}

class Logger:
    def __init__(self, log_path, exp_name, save=True, checkpoint=None, resume=False, finetune=False):
        self.save = save
        self.resume = resume
        self.finetune = finetune
        self.checkpoint = checkpoint
        self.exp_path = None
        self.log_time = None

        # Resolve experiment path and training mode
        init_log = []
        base_exp_dir = os.path.join(log_path, exp_name)

        # Case 1: Explicit checkpoint path provided
        if self.checkpoint:
            if not os.path.exists(self.checkpoint):
                raise FileNotFoundError(f"[Logger]  Checkpoint file not found: {self.checkpoint}")
                
            self.exp_path = os.path.dirname(self.checkpoint)
            self.log_time = os.path.basename(self.exp_path)
            init_log.append(f"[Logger]  Loading explicit checkpoint: {self.checkpoint}")
            
            if self.finetune:
                self.exp_path = os.path.join(self.exp_path, "finetune")
                init_log.append("[Logger]  Entering finetune mode - creating finetune subdirectory")
                
        # Case 2: Resume or finetune without explicit checkpoint
        elif self.resume or self.finetune:
            if not os.path.exists(base_exp_dir):
                raise FileNotFoundError(f"[Logger]  Experiment directory not found: {base_exp_dir}")
                
            # Find latest experiment
            subdirs = [d for d in sorted(os.listdir(base_exp_dir)) 
                      if os.path.isdir(os.path.join(base_exp_dir, d))]
            
            if not subdirs:
                raise FileNotFoundError(f"[Logger]  No existing runs found in {base_exp_dir}")
                
            self.log_time = subdirs[-1]
            self.exp_path = os.path.join(base_exp_dir, self.log_time)
            
            if self.finetune:
                self.exp_path = os.path.join(self.exp_path, "finetune")
                init_log.append(f"[Logger]  Starting finetune from {self.log_time}")
            else:
                init_log.append(f"[Logger]  Resuming training from {self.log_time}")
                
        # Case 3: Fresh training
        else:
            self.log_time = time.strftime('%Y-%m-%d_%H-%M-%S')
            self.exp_path = os.path.join(base_exp_dir, self.log_time)
            init_log.append(f"[Logger]  Starting new experiment: {self.log_time}")

        # Create directory structure
        os.makedirs(self.exp_path, exist_ok=True)
        init_log.append(f"[Logger]  Experiment path: {self.exp_path}")

        # Set up log file
        self.log_file = os.path.join(self.exp_path, "log.txt")
        self.metrics = Metrics()

        # Write initial logs
        self.logline()
        for txt in init_log:
            self.log(txt)

    def log(self, message):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        lines = message.split('\n')
        for line in lines:
            print(line, flush=True)
            if self.save:
                with open(self.log_file, 'a') as f:
                    print(f"# {current_time} # {line}", flush=True, file=f)

    def log_config(self, config):
        self.log("[Logger]  Configuration:")
        max_key_length = max(len(key) for key in config.keys())
        for key, value in config.items():
            self.log(f"\t -> {key.ljust(max_key_length)}: {value}")

    def logline(self):
        self.log('\n' + '-' * 50 + '\n')

    def logsubline(self):
        self.log('\n' + '-' * 50 + '\n')

    def reset_metrics(self, interval=False):
        if interval:
            self.metrics.reset_interval()
        else:
            self.metrics.reset()

    def update_metrics(self, key, value, count=1):
        self.metrics.update(key, value, count)

    def log_step(self, epoch, step, total_steps, training, time_elapsed, interval=False):
        if training:
            self.log(
                f"[Epoch {epoch} | Train Step {step}/{total_steps}] "
                f"Loss | G: {self.metrics.average('g_loss', interval):.4f}, D: {self.metrics.average('d_loss', interval):.4f} || "
                f"Grad Norm | G: {self.metrics.average('g_grad_norm', interval):.4f}, D: {self.metrics.average('d_grad_norm', interval):.4f} || "
                f"LR | G: {sci(self.metrics.average('g_lr', interval))}, D: {sci(self.metrics.average('d_lr', interval))} || "
                f"Time: {time_elapsed:.2f}s (Avg: {self.metrics.average('step_time', interval):.2f}s/Step)"
            )
        else:
            self.log(
                f"[Epoch {epoch} | Valid Step {step}/{total_steps}] "
                f"Loss | G: {self.metrics.average('g_loss', interval):.4f} || "
                f"Time: {time_elapsed:.2f}s (Avg: {self.metrics.average('step_time', interval):.2f}s/Step)"
            )

    def log_epoch(self, epoch, epoch_duration, best_g_loss, best_epoch):
        self.log(
            f"\n{'-'*50}\n"
            f"[Epoch {epoch}]:\n"
            f"\tTrain Loss | G: {self.metrics.average('train_g_loss'):.4f}, D: {self.metrics.average('train_d_loss'):.4f}\n"
            f"\tValid Loss | G: {self.metrics.average('val_g_loss'):.4f}\n"
            f"\tBest  Loss | G: {best_g_loss:.4f} (epoch {best_epoch})\n"
            f"\tTime       | {epoch_duration}\n"
            f"{'-'*50}\n"
        )