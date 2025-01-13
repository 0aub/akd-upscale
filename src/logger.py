import time
import os

class Logger:
    def __init__(self, log_path, exp_name, save=True):
        self.save = save
        self.log_time = time.strftime('%Y-%m-%d_%H-%M-%S')
        self.exp_path = os.path.join(log_path, exp_name, self.log_time)

        if not os.path.exists(self.exp_path):
            os.makedirs(self.exp_path)

        self.log_file = os.path.join(self.exp_path, "log.txt")

    def log(self, message):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        lines = message.split('\n')
        for line in lines:
            print(line, flush=True)
            if self.save:
                with open(self.log_file, 'a') as f:
                    print(f"# {current_time} # {line}", flush=True, file=f)

    def log_config(self, config):
        self.log("[INFO]  Configuration:")
        max_key_length = max(len(key) for key in config.keys())
        for key, value in config.items():
            self.log(f"\t -> {key.ljust(max_key_length)}: {value}")

    def logline(self):
        self.log('\n' + '='*100 + '\n')
