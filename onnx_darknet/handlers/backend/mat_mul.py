# import tensorflow as tf

from onnx_darknet.handlers.backend_handler import BackendHandler
from onnx_darknet.handlers.handler import onnx_op
from onnx_darknet.handlers.handler import darknet_func


@onnx_op("MatMul")
@darknet_func(tf.matmul)
class MatMul(BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return [cls.make_tensor_from_onnx_node(node, **kwargs)]
