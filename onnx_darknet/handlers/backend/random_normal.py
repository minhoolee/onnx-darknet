# import tensorflow as tf

from onnx_darknet.handlers.backend_handler import BackendHandler
from onnx_darknet.handlers.handler import onnx_op
from onnx_darknet.handlers.handler import darknet_func


@onnx_op("RandomNormal")
@darknet_func(tf.random_normal)
class RandomNormal(BackendHandler):

  @classmethod
  def get_attrs_processor_param(cls):
    return {"default": {"mean": 0., "scale": 1.}, "rename": {"scale": "stddev"}}

  @classmethod
  def version_1(cls, node, **kwargs):
    return [cls.make_tensor_from_onnx_node(node, **kwargs)]
