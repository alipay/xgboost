from collections import OrderedDict
from typing import NamedTuple


def load_config(cls, **kwargs):
    """
    Load (initialize) config structures (similar to NamedTuple,
    which contains attr('_field_types') from meta class and raw configs (kv-style)).
    This method runs recursively to build nested structures from nested param dicts.
    And param-specific convert functions are supported to parse various kinds of inputs.

    Below is an example.

    class Field(NamedTuple):
        a: int
        b: bool = False
        c: str = ''

        @classmethod
        def convert_c(cls, value):
            return '@'.join(value)

    field = load_config(Field, 'a' = 1, 'c' = ['a', 'b', 'c'])
    assert field.c == 'a@b@c'
    assert field.b == False
    assert field.a == 1


    :param cls: meta class of config
    :param kwargs: raw config
    :return: an instance of config structures
    """

    # TODO: print more exception details when raising an Error
    assert isinstance(kwargs, dict)
    assert hasattr(cls, '_field_types')
    tpe = cls._field_types
    config = {}
    for k, v in kwargs.items():
        # skip None values
        if v is None:
            continue
        if hasattr(tpe[k], '_field_types'):
            if hasattr(cls, 'convert_' + k):
                convert_fn = getattr(cls, 'convert_' + k)
                v = convert_fn(v)
            # skip empty kwargs, using default value
            if not v:
                continue
            assert isinstance(v, dict), 'Try to init class(%s) with %s!' % (tpe[k], v)
            config[k] = load_config(tpe[k], **v)
        else:
            assert k in tpe, 'Unexpected key %s in %s!' % (k, cls)
            if hasattr(cls, 'convert_' + k):
                convert_fn = getattr(cls, 'convert_' + k)
                config[k] = convert_fn(v)
            else:
                config[k] = v

    return cls(**config)


def dump_config(fields: NamedTuple):
    """
    Convert config structures back to raw configs.

    :param fields: the config instance to be dumped
    :return: raw configs, a python dict
    """

    assert hasattr(fields, '_field_types')
    config = {}
    for k, v in fields._asdict().items():
        if v is None:
            continue
        if hasattr(v, '_field_types'):
            config[k] = dump_config(v)
        else:
            config[k] = v
    return config


def field_keys_equal(a, b):
    """
    Check equality of `field_keys` recursively between two instances who contains `field_types`.

    This method is widely used to take place of "isinstance(config instance, config meta)",
    which sometimes returns unexpected False.

    :param a:
    :param b:
    :return: whether keys of a are fully equal to keys of b
    """

    assert hasattr(a, '_field_types')
    assert hasattr(b, '_field_types')
    assert isinstance(a._field_types, OrderedDict)
    assert isinstance(b._field_types, OrderedDict)
    for k, v in a._field_types.items():
        if k not in b._field_types:
            return False
        if hasattr(v, '_field_types'):
            if not hasattr(b._field_types[k], '_field_types'):
                return False
            eq = field_keys_equal(v, b._field_types[k])
            if not eq:
                return False
    return True


def fields_equal(a, b):
    """
    Check fields equality between two config structures recursively.

    :param a:
    :param b:
    :return: whether a and b have exactly same fields
    """

    assert hasattr(a, '_field_types')
    assert hasattr(b, '_field_types')
    assert isinstance(a._field_types, OrderedDict)
    assert isinstance(b._field_types, OrderedDict)
    for k in a._field_types.keys():
        if k not in b._field_types:
            return False
        v = getattr(a, k)
        if hasattr(v, '_field_types'):
            if not hasattr(b._field_types[k], '_field_types'):
                return False
            eq = fields_equal(v, b._field_types[k])
            if not eq:
                return False
        if v != getattr(b, k):
            return False
    return True
