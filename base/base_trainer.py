import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
from utils import global_var

class BaseTrainer:
    def __init__(self, model, criterion, metric_ftns, optimizer, config):
        """
        @param model: 实例化的torch模型，模型的类写在model.model文件下
        @param criterion: 自己编写损失函数的函数句柄，
        @param metric_ftns: 为list of 函数句柄，这里的函数句柄计算的数据会在tensorboard scalar中显示，函数写在model.metric下
        @param optimizer: torch.optim的诸多优化器类中的一个实例
        @param config: 实例化的parse_config.ConfugParser类
        """
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')
        self.save_non_optimum = cfg_trainer['save_non_optimum']

        if self.save_non_optimum:
            # 如果保存非最佳结果，那么根据save_period判断几个epoch保存一次
            assert self.save_period is not None, 'since save_non_optimum is true, save_period should not be None'
        else:
            # 如果不保存非最佳，那么save_period应为none
            assert self.save_period is None, 'since save_non_optimum is false, save_period should be None'

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance                
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        写了整个训练过程中的逻辑，其实就是重复运行单步训练逻辑，单步训练逻辑写在trainer.trainer.Trainer类中的方法_train_epoch中
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)  # 运行单步

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))
                global_var.get_value('email_log').add_log('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning(
                        f"Warning: Metric '{self.mnt_metric}' is not found. Model performance monitoring is disabled.")
                    global_var.get_value('email_log').add_log(
                        f"Warning: Metric '{self.mnt_metric}' is not found. Model performance monitoring is disabled.")
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(
                        f"Validation performance didn\'t improve for {self.early_stop} epochs. Training stops.")
                    global_var.get_value('email_log').add_log(
                        f"Validation performance didn\'t improve for {self.early_stop} epochs. Training stops.")
                    break

            if best:
                self._save_best_checkpoint(epoch)

            if self.save_non_optimum:
                if epoch % self.save_period == 0:
                    self._save_non_optimum_opcheckpoint(epoch)

    def _save_non_optimum_opcheckpoint(self, epoch):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info(f"Saving checkpoint: {filename} ...")
        global_var.get_value('email_log').add_log(f"Saving checkpoint: {filename} ...")

    def _save_best_checkpoint(self, epoch):
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        best_path = str(self.checkpoint_dir / 'model_best.pth')
        torch.save(state, best_path)
        self.logger.info("Saving current best: model_best.pth ...")
        global_var.get_value('email_log').add_log("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        global_var.get_value('email_log').add_log(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
            global_var.get_value('email_log').add_log("Warning: Architecture configuration given in config file is different from that of checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
            global_var.get_value('email_log').add_log("Warning: Optimizer type given in config file is different from that of checkpoint. Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info(f"Checkpoint loaded. Resume training from epoch {self.start_epoch}")
        global_var.get_value('email_log').add_log(f"Checkpoint loaded. Resume training from epoch {self.start_epoch}")
