from onnx_darknet.common import exception
from onnx_darknet.handlers.frontend_handler import FrontendHandler
from onnx_darknet.handlers.handler import onnx_op
from onnx_darknet.handlers.handler import darknet_op
from .conv_mixin import ConvMixin


@onnx_op("Conv")
@darknet_op(["Conv1D", "Conv2D", "Conv3D"])
class Convolution(ConvMixin, FrontendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    if node.op_type == "Conv1D":
      d = 1
    elif node.op_type == "Conv2D":
      d = 2
    elif node.op_type == "Conv3D":
      d = 3
    else:
      exception.OP_UNSUPPORTED_EXCEPT(node.op_type, "Tensorflow")
    return cls.conv_op(node, d=d, **kwargs)
