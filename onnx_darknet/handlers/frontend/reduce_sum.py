from onnx_darknet.handlers.frontend_handler import FrontendHandler
from onnx_darknet.handlers.handler import onnx_op
from onnx_darknet.handlers.handler import darknet_op
from .math_mixin import ReductionMixin


@onnx_op("ReduceSum")
@darknet_op("Sum")
class ReduceSum(ReductionMixin, FrontendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.reduction_op(node, **kwargs)
