from onnx_darknet.handlers.frontend_handler import FrontendHandler
from onnx_darknet.handlers.handler import onnx_op
from onnx_darknet.handlers.handler import darknet_op
from .control_flow_mixin import LogicalMixin


@onnx_op("Not")
@darknet_op("LogicalNot")
class LogicalNot(LogicalMixin, FrontendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.logical_op(node, broadcast=0, **kwargs)
