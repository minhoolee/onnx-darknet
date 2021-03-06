from onnx_darknet.handlers.frontend_handler import FrontendHandler
from onnx_darknet.handlers.handler import onnx_op
from onnx_darknet.handlers.handler import darknet_op
from .math_mixin import BasicMathMixin


@onnx_op("Pow")
@darknet_op("Pow")
class Pow(BasicMathMixin, FrontendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.basic_math_op(node, **kwargs)

  @classmethod
  def version_7(cls, node, **kwargs):
    return cls.basic_math_op(node, **kwargs)
