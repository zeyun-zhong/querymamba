"""Implementation of metrictracker in training process"""
from typing import Dict, Union, Optional
import torch
import torch.distributed as dist

from .distributed import is_dist_avail_and_initialized, all_gather
from .logging import get_logger


logger = get_logger(__name__)


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name, string_format='{:.3f}', device="cuda"):
        self.name = name
        self.string_format = string_format
        self.device = device

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        count_all = torch.tensor(self.count, device=self.device)
        sum_all = torch.tensor(self.sum, device=self.device)
        dist.barrier()
        dist.all_reduce(count_all)
        dist.all_reduce(sum_all)
        self.count = count_all
        self.sum = sum_all

    def value(self, idx: Optional[Union[int, str]] = -1):
        """returns the current floating average"""
        del idx
        self.avg = self.sum / self.count
        return self.avg

    def to_string(self, idx: Optional[int] = -1):
        del idx  # Not used here.
        return self.string_format.format(self.value())


class MetricTracker:
    """Interface of all metrics, tracks multiple metrics"""
    def __init__(
            self,
            device="cuda",
            **kwargs,
    ):
        self.training_metrics = {}
        self.validation_metrics = {}
        self.training_prefix = 'train/'
        self.validation_prefix = 'val/'
        self.kwargs = kwargs
        self.device = device

    def add_metric(self, name, is_training=None, metric_class=None):
        if metric_class is not None:
            meter = globals()[metric_class](
                name, device=self.device, **self.kwargs)
        else:
            meter = AverageMeter(name, device=self.device)

        logger.info(f'Added {type(meter).__name__} for the metric {name}.')

        # reset the meter
        meter.reset()

        if is_training is None:
            self.training_metrics[name] = meter
            self.validation_metrics[name] = meter
        elif is_training:
            self.training_metrics[name] = meter
        else:
            self.validation_metrics[name] = meter

    def update(self, metric_dict: Dict, batch_size: int, is_training: bool):
        if is_training:
            metrics = self.training_metrics
            prefix = self.training_prefix
        else:
            metrics = self.validation_metrics
            prefix = self.validation_prefix

        for key, value in metric_dict.items():
            key = prefix + key
            metric_cls = None
            counts = batch_size

            if isinstance(value, (list, tuple)):
                assert len(value) == 3, \
                    (f"Metric {key} should contain three elements: "
                     f"Metric class name, contents to be updated, and counts")
                metric_cls, value, tmp = value
                if tmp is not None:
                    counts = tmp

            if key not in metrics:
                self.add_metric(key, is_training, metric_class=metric_cls)
            metrics[key].update(value, counts)

    def synchronize_between_processes(self, is_training):
        if is_training:
            metrics = self.training_metrics
        else:
            metrics = self.validation_metrics

        for key in metrics:
            metrics[key].synchronize_between_processes()

    def reset(self):
        """reset all metrics at the beginning of each training epoch"""
        for name in self.training_metrics:
            self.training_metrics[name].reset()
        for name in self.validation_metrics:
            self.validation_metrics[name].reset()

    def get_all_data(self, is_training, idx: Optional[Union[int, str]] = -1):
        """returns the current values of all tracked metrics"""
        if is_training:
            metrics = self.training_metrics
        else:
            metrics = self.validation_metrics
        data = {}
        for key in metrics:
            data[key] = metrics[key].value(idx)
        return data

    def get_data(self, metric_name, is_training,
                 idx: Optional[Union[int, str]] = -1):
        """returns the current value of the metric"""
        if is_training:
            return self.training_metrics[metric_name].value(idx)
        else:
            return self.validation_metrics[metric_name].value(idx)

    def to_string(self, is_training, idx: Optional[Union[int, str]] = -1):
        """returns the string of all values"""
        if is_training:
            result = 'Training:    '  # '\33[0;36;40m' + 'Training:    '
            metrics = self.training_metrics
        else:
            result = 'Validation:  '  # '\33[0;32;40m' + 'Validation:  '
            metrics = self.validation_metrics

        for key in metrics:
            result += metrics[key].name + ': ' + metrics[key].to_string(idx) + '   '
        return result   # + '\033[0m'
