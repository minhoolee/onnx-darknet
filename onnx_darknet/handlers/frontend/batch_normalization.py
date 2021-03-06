from onnx_darknet.handlers.frontend_handler import FrontendHandler
from onnx_darknet.handlers.handler import onnx_op
from onnx_darknet.handlers.handler import darknet_op


@onnx_op("BatchNormalization")
@darknet_op(["BatchNorm", "FusedBatchNorm"])
class BatchNormalization(FrontendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls.make_node_from_tf_node(
        node,
        epsilon=node.attr.get("epsilon", 1e-5),
        is_test=node.attr.get("is_training", 0),
        consumed_inputs=node.attr.get("consumed_inputs", [0, 0, 0, 1, 1]))

  @classmethod
  def version_6(cls, node, **kwargs):
    return cls.make_node_from_tf_node(
        node,
        epsilon=node.attr.get("epsilon", 1e-5),
        is_test=node.attr.get("is_training", 0))

  @classmethod
  def version_7(cls, node, **kwargs):
    return cls.make_node_from_tf_node(
        node, epsilon=node.attr.get("epsilon", 1e-5))
