from onnx_darknet.handlers.frontend_handler import FrontendHandler
from onnx_darknet.handlers.handler import onnx_op
from onnx_darknet.handlers.handler import darknet_op


@onnx_op("Relu")
@darknet_op("Relu")
class Relu(FrontendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.make_node_from_tf_node(node)

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls.make_node_from_tf_node(node)
