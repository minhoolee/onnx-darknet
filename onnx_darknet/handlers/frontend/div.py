from onnx_darknet.handlers.frontend_handler import FrontendHandler
from onnx_darknet.handlers.handler import onnx_op
from onnx_darknet.handlers.handler import darknet_op
from .math_mixin import ArithmeticMixin


@onnx_op("Div")
@darknet_op("RealDiv")
class Div(ArithmeticMixin, FrontendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.arithmetic_op(node, **kwargs)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls.arithmetic_op(node, **kwargs)

  @classmethod
  def version_7(cls, node, **kwargs):
    return cls.arithmetic_op(node, **kwargs)
