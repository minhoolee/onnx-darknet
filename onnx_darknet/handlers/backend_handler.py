from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import inspect

# import tensorflow as tf

from onnx_darknet.common import IS_PYTHON3
from onnx_darknet.common import get_data_format
from onnx_darknet.common import get_perm_from_formats
from onnx_darknet.common import supports_device
from .handler import Handler


class BackendHandler(Handler):
  """ This class is base backend handler class.
  All backend operator handler class MUST inherit this class.
  In backend, operator handler class's name should be pascal case of file name
  which should be snake case.
  Use ONNX operator name as class name.
  """

  DARKNET_FUNC = None

  @classmethod
  def get_attrs_processor_param(cls):
    """ Get param for attrs processor.

    :return: Dict.
    """
    return {}

  @classmethod
  def _process_attrs(cls, attrs):
    """ Private method for processing attrs.
    Param for this processor got from `get_attrs_processor_param`.
    Param is dict contains two key: `default` and `rename`.
    First add default value to attrs if key does not exist.
    Second rename key to new key.

    For example:
      attrs = {"keep_dims": True}
      param = {"default": {"axis": 1},
               "rename": {"keep_dims": "keepdims"}}

      processed_attrs = {"axis": "1", "keepdims": True}

    :param attrs: Process target attrs.
    :return: Processed attrs.
    """
    param = {"rename": {}, "default": {}}
    param.update(cls.get_attrs_processor_param())

    for k, v in param["default"].items():
      attrs.setdefault(k, v)

    for k, new_k in param["rename"].items():
      if k in attrs:
        attrs[new_k] = attrs.pop(k)

    return attrs

  @classmethod
  def make_tensor_from_onnx_node(cls,
                                 node,
                                 darknet_func=None,
                                 inputs=None,
                                 attrs=None,
                                 name="",
                                 c_first_cuda_only=False,
                                 c_last_only=False,
                                 **kwargs):
    """ Helper method to make tensor.

    :param node: OnnxNode object.
    :param darknet_func: Callable Darknet function. Default is cls.DARKNET_FUNC.
    :param inputs: Inputs tensor. Default is got from node.inputs.
    :param attrs: Attributes. Default is node.attrs.
    :param name: Node name.
    :param c_first_cuda_only: If channel first is only supported by cuda.
    If true and not cuda, do pre and post transpose.
    :param c_last_only: If only channel last is support,
    do pre and post transpose.
    :param kwargs: Other args.
    :return: Tensor.
    """
    tensor_dict = kwargs.get("tensor_dict", {})
    darknet_func = darknet_func or cls.DARKNET_FUNC
    if darknet_func is None:
      raise RuntimeError("No Darknet function is given.")
    if inputs is None:
      inputs = [tensor_dict.get(inp, None) for inp in node.inputs]
    if attrs is None:
      attrs = copy.deepcopy(node.attrs)
    name = name or node.name
    if name != "":
      attrs["name"] = name

    if c_first_cuda_only and c_last_only:
      raise ValueError(
          "c_first_cuda_only and c_last_only can not both be True.")

    if c_first_cuda_only:
      return cls.c_first_cuda_only(darknet_func, inputs, attrs)
    elif c_last_only:
      return cls.c_last_only(darknet_func, inputs, attrs)

    return cls._run_darknet_func(darknet_func, inputs, attrs)

  @classmethod
  def c_first_cuda_only(cls, darknet_func, inputs, attrs):
    """ Handle operator that channel first is only supported by CUDA.
    When using CPU, two transposes should be added.

    :param darknet_func: Callable Darknet function.
    :param inputs: Inputs tensor.
    :param attrs: Attributes.
    :return: Tensor.
    """
    support_cuda = supports_device("CUDA")
    if not support_cuda:
      return cls._tuck_transpose(darknet_func, inputs, attrs)
    return cls._run_darknet_func(darknet_func, inputs, attrs)

  @classmethod
  def c_last_only(cls, darknet_func, inputs, attrs):
    """ Handle operator that channel last only is supported.
    Add two transposes anyway.

    :param darknet_func: Callable Darknet function.
    :param inputs: Inputs tensor.
    :param attrs: Attributes.
    :return: Tensor.
    """
    storage_format, compute_format = get_data_format(len(inputs[0].get_shape()))
    compute_format = compute_format.replace("C", "") + "C"
    return cls._tuck_transpose(darknet_func, inputs, attrs,
                               (storage_format, compute_format))

  @classmethod
  def _tuck_transpose(cls, darknet_func, inputs, attrs, data_format=None):
    x = inputs[0]
    x_rank = len(x.get_shape())
    if not data_format:
      data_format = get_data_format(x_rank)
    pre_perm = get_perm_from_formats(data_format[0], data_format[1])
    post_perm = get_perm_from_formats(data_format[1], data_format[0])
    attrs["data_format"] = data_format[1]
    if pre_perm != list(range(x_rank)):
      x_t = tf.transpose(x, perm=pre_perm)
      y = cls._run_darknet_func(darknet_func, [x_t] + inputs[1:], attrs)
      y_t = tf.transpose(y, perm=post_perm)
      return y_t
    return cls._run_darknet_func(darknet_func, inputs, attrs)

  @classmethod
  def _run_darknet_func(cls, darknet_func, inputs, attrs):
    """ Run Darknet function.
    Use only acceptable attributes of function from attrs.

    :param darknet_func: Darknet function.
    :param inputs: Inputs.
    :param attrs: Attributes.
    :return: Tensor.
    """
    if IS_PYTHON3:
      params = list(inspect.signature(darknet_func).parameters.keys())
    else:
      # use closure to get args for function using decorator
      if darknet_func.__closure__ is not None:
        params = darknet_func.__closure__[1].cell_contents.args
      else:
        params = inspect.getargspec(darknet_func).args

    attrs = cls._process_attrs(attrs)
    return darknet_func(*inputs,
                   **dict([(p, attrs[p]) for p in params if p in attrs]))
