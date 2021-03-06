# import tensorflow as tf

from onnx_darknet.handlers.backend_handler import BackendHandler
from onnx_darknet.handlers.handler import onnx_op
from onnx_darknet.handlers.handler import darknet_func
from .math_mixin import ReductionMixin


@onnx_op("ReduceProd")
@darknet_func(tf.reduce_prod)
class ReduceProd(ReductionMixin, BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)
