import os
import logging
from pathlib import Path
from tensorboardX import SummaryWriter
from queue import PriorityQueue
import torch

def create_callbacks(name, dumps):
    log_dir = Path(dumps['path']) / dumps['logs'] / name
    model_dir = Path(dumps['path']) / dumps['weights'] / name
    callbacks = Callbacks(
        [
            Logger(log_dir),
            TensorBoard(str(log_dir)),
            CheckpointSaver(
                save_dir=model_dir,
                save_name='epoch_{epoch}.pth',
                num_checkpoints=4,
                mode='max'
            )
        ]
    )
    # callbacks = Callbacks([Logger(log_dir), TensorBoard(str(log_dir))])
    return callbacks


class Callback(object):
    def __init__(self):
        self.runner = None

    def set_runner(self, runner):
        self.runner = runner

    def on_batch_begin(self, i, **kwargs):
        pass

    def on_batch_end(self, i, **kwargs):
        pass

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass


class Callbacks(Callback):
    def __init__(self, callbacks):
        super().__init__()
        if isinstance(callbacks, Callbacks):
            self.callbacks = callbacks.callbacks
        if isinstance(callbacks, list):
            self.callbacks = callbacks
        else:
            self.callbacks = []

    def set_runner(self, runner):
        super().set_runner(runner)
        for callback in self.callbacks:
            callback.set_runner(runner)

    def on_batch_begin(self, i, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_begin(i, **kwargs)

    def on_batch_end(self, i, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_end(i, **kwargs)

    def on_epoch_begin(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch)

    def on_epoch_end(self, epoch, epoch_info):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, epoch_info)

    def on_train_begin(self):
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end()


class Logger(Callback):
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
        self.logger = None

    def on_train_begin(self):
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = self._get_logger(str(self.log_dir / 'logs.txt'))
        self.logger.info(f'Starting training with params:\n{self.runner.params}\n\n')
        self.logger.info(self.runner.model)
        # if os.path.isfile(self.runner.weight):
        #     self.logger.info("=>loading weight '{}' \n".format(self.runner.weight))

    def on_epoch_begin(self, epoch):
        self.logger.info(
            f'Epoch {epoch} | '
            f'optimizer "{self.runner.optimizer.__class__.__name__}" | '
            f'lr {self.current_lr}'
        )

    def on_epoch_end(self, epoch, epoch_info):
        train_metrics, val_metrics = {}, {}
        for k, v in epoch_info.items():
            train_metrics[k] = v['train'][epoch]
            val_metrics[k] = v['val'][epoch]
        self.logger.info('Train metrics: ' + self._get_metrics_string(train_metrics))
        self.logger.info('Valid metrics: ' + self._get_metrics_string(val_metrics) + '\n')
    
    @staticmethod
    def _get_logger(log_path):
        logger = logging.getLogger(log_path)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
        formatter = logging.Formatter(fmt)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger

    @property
    def current_lr(self):
        res = []
        for param_group in self.runner.optimizer.param_groups:
            res.append(param_group['lr'])
        if len(res) == 1:
            return res[0]
        return res

    @staticmethod
    def _get_metrics_string(metrics):
        return ' | '.join(k+': {:.5f}'.format(v) for k, v in metrics.items())


class TensorBoard(Callback):
    def __init__(self, log_dir: str):
        super().__init__()
        self.log_dir = log_dir
        self.writer = None

    def on_train_begin(self):
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer_train = SummaryWriter(os.path.join(self.log_dir, 'train'))
        self.writer_val = SummaryWriter(os.path.join(self.log_dir, 'val'))

    # def on_batch_end(self, i, **kwargs):
        
    def on_epoch_end(self, epoch, epoch_info):
        train_metrics, val_metrics = {}, {}
        for k, v in epoch_info.items():
            train_metrics[k] = v['train'][epoch]
            val_metrics[k] = v['val'][epoch]

        for k, v in train_metrics.items():
            self.writer_train.add_scalar(f'{k}', float(v), global_step=epoch)

        for k, v in val_metrics.items():
            self.writer_val.add_scalar(f'{k}', float(v), global_step=epoch)

        for idx, param_group in enumerate(self.runner.optimizer.param_groups):
            lr = param_group['lr']
            self.writer_train.add_scalar(f'group{idx}/lr', float(lr), global_step=epoch)

    def on_train_end(self):
        self.writer_train.close()
        self.writer_val.close()


class CheckpointSaver(Callback):
    def __init__(self, save_dir, save_name, num_checkpoints, mode):
        super().__init__()
        self.mode = mode
        self.save_name = save_name
        self._best_checkpoints_queue = PriorityQueue(num_checkpoints)
        # self._metric_name = metric_name
        self.save_dir = save_dir

    def on_train_begin(self):
        os.makedirs(self.save_dir, exist_ok=True)

    def save_checkpoint(self, epoch, path):
        if hasattr(self.runner.model, 'module'):
            state_dict = self.runner.model.module.state_dict()
        else:
            state_dict = self.runner.model.state_dict()
        torch.save(state_dict, path)

    def on_epoch_end(self, epoch, epoch_info):
        train_metrics, val_metrics = {}, {}
        for k, v in epoch_info.items():
            train_metrics[k] = v['train'][epoch]
            val_metrics[k] = v['val'][epoch]

        metric = val_metrics['dice']
        new_path_to_save = os.path.join(
            self.save_dir,
            self.save_name.format(epoch=epoch, metric='{:.5}'.format(metric))
        )
        if self._try_update_best_losses(metric, new_path_to_save):
            self.save_checkpoint(epoch=epoch, path=new_path_to_save)

    def _try_update_best_losses(self, metric, new_path_to_save):
        if self.mode == 'min':
            metric = -metric
        if not self._best_checkpoints_queue.full():
            self._best_checkpoints_queue.put((metric, new_path_to_save))
            return True

        min_metric, min_metric_path = self._best_checkpoints_queue.get()

        if min_metric < metric:
            os.remove(min_metric_path)
            self._best_checkpoints_queue.put((metric, new_path_to_save))
            return True

        self._best_checkpoints_queue.put((min_metric, min_metric_path))
        return False