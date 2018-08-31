# import tensorflow as tf

from onnx_darknet.handlers.backend_handler import BackendHandler
from onnx_darknet.handlers.handler import onnx_op
from onnx_darknet.handlers.handler import darknet_func
from .math_mixin import ArithmeticMixin


@onnx_op("Sum")
@darknet_func(tf.add_n)
class Sum(ArithmeticMixin, BackendHandler):

  @classmethod
  def _common(cls, node, **kwargs):
    tensor_dict = kwargs["tensor_dict"]
    return [
        cls.make_tensor_from_onnx_node(
            node,
            inputs=[[tensor_dict.get(inp, None) for inp in node.inputs]],
            **kwargs)
    ]

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_8(cls, node, **kwargs):
    return cls._common(node, **kwargs)
