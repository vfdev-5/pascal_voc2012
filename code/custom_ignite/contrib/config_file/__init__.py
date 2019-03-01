import random
from functools import partial

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path

import attr
from attr.validators import instance_of


@attr.s
class _BaseConfig(object):

    def asdict(self):
        return attr.asdict(self)

    def __contains__(self, attrib_name):
        return hasattr(self, attrib_name)


@attr.s
class BaseConfig(_BaseConfig):

    seed = attr.ib(default=attr.Factory(partial(random.randint, a=0, b=1000)), validator=instance_of(int))
    device = attr.ib(default='cpu', validator=instance_of(str))
    debug = attr.ib(default=False, validator=instance_of(bool))


def is_iterable_with_length(instance, attribute, value):
    if not (hasattr(value, "__len__") and hasattr(value, "__iter__")):
        raise TypeError("Argument '{}' should be iterable with length".format(attribute.name))


def is_positive(instance, attribute, value):
    if value < 1:
        raise ValueError("Argument '{}' should be positive".format(attribute.name))


def is_dict_of_key_value_type(key_type, value_type):
    def _validator(instance, attribute, value):
        if not isinstance(value, dict) or len(value) == 0:
            raise TypeError("Argument '{}' should be non-empty dictionary".format(attribute.name))

        if not all([isinstance(k, key_type) and isinstance(v, value_type)
                    for k, v in value.items()]):
            raise ValueError("Argument '{}' should be dictionary of ".format(attribute.name) +
                             "keys of type '{}' and values of type '{}'".format(key_type, value_type))

    return _validator


def is_callable(instance, attribute, value):
    if not (callable(value)):
        raise TypeError("Argument '{}' should be callable".format(attribute.name))
