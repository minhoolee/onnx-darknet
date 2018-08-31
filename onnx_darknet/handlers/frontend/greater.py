from onnx_darknet.handlers.frontend_handler import FrontendHandler
from onnx_darknet.handlers.handler import onnx_op
from onnx_darknet.handlers.handler import darknet_op
from .control_flow_mixin import ComparisonMixin


@onnx_op("Greater")
@darknet_op("Greater")
class Greater(ComparisonMixin, FrontendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.comparison_op(node, **kwargs)

  @classmethod
  def version_7(cls, node, **kwargs):
    return cls.comparison_op(node, **kwargs)
