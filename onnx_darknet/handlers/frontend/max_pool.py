from onnx_darknet.handlers.frontend_handler import FrontendHandler
from onnx_darknet.handlers.handler import onnx_op
from onnx_darknet.handlers.handler import darknet_op
from .pool_mixin import PoolMixin


@onnx_op("MaxPool")
@darknet_op("MaxPool")
class MaxPool(PoolMixin, FrontendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.pool_op(node, **kwargs)

  @classmethod
  def version_8(cls, node, **kwargs):
    return cls.pool_op(node, **kwargs)
