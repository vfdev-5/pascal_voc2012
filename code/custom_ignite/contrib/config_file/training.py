import attr
from attr.validators import optional, instance_of, and_

import torch
import torch.nn as nn
from torch.optim import Optimizer

from ignite.metrics import Metric

from custom_ignite.contrib.config_file import _BaseConfig, is_dict_of_key_value_type, is_positive, is_iterable_with_length


__all__ = ['ModelConfig', 'SolverConfig', 'ValidationConfig']


# @attr.s
# class LoggingConfig(_BaseConfig):
#     """Logging configuration

#     Args:
#         log_interval (int): logging interval in number of iterations. Logging will happen every `log_interval`
#             iterations.
#         checkpoint_interval (int): 
#     """

#     log_interval = attr.ib(default=100, validator=optional(and_(instance_of(int), is_positive)))

#     checkpoint_interval = attr.ib(default=1000, validator=optional(and_(instance_of(int), is_positive)))


@attr.s
class ModelConfig(_BaseConfig):
    """

    """
    model = attr.ib(default=nn.Module(), validator=instance_of(nn.Module))

    weights_filepath = attr.ib(default=None, validator=optional(instance_of(str)))


_dummy_optim = torch.optim.Optimizer([torch.Tensor(0)], {})


@attr.s
class SolverConfig(_BaseConfig):
    """

    """

    optimizer = attr.ib(default=_dummy_optim, validator=instance_of(Optimizer))

    criterion = attr.ib(default=nn.Module(), validator=instance_of(nn.Module))

    num_epochs = attr.ib(default=None, validator=optional(and_(instance_of(int), is_positive)))
    num_iterations = attr.ib(default=None, validator=optional(and_(instance_of(int), is_positive)))


class _DummyMetric(Metric):

    def reset(self, *args, **kwargs): pass

    def compute(self, *args, **kwargs): pass

    def update(self, *args, **kwargs): pass


_dummy_metric = {"d": _DummyMetric()}


@attr.s
class ValidationConfig(_BaseConfig):

    val_dataloader = attr.ib(default=[], validator=is_iterable_with_length)

    val_metrics = attr.ib(default=_dummy_metric, validator=is_dict_of_key_value_type(str, Metric))
    
    model_checkpoint_kwargs = attr.ib(default=None, validator=optional(instance_of(dict)))

    train_metrics = attr.ib(default=None, validator=optional(is_dict_of_key_value_type(str, Metric)))

    train_eval_dataloader = attr.ib(default=None, validator=optional(is_iterable_with_length))
