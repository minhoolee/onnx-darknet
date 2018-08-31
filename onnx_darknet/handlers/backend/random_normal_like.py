# import tensorflow as tf

from onnx_darknet.handlers.backend_handler import BackendHandler
from onnx_darknet.handlers.handler import onnx_op
from onnx_darknet.handlers.handler import darknet_func


@onnx_op("RandomNormalLike")
@darknet_func(tf.random_normal)
class RandomNormalLike(BackendHandler):

  @classmethod
  def get_attrs_processor_param(cls):
    return {"default": {"mean": 0., "scale": 1.}, "rename": {"scale": "stddev"}}

  @classmethod
  def version_1(cls, node, **kwargs):
    inputs = [kwargs["tensor_dict"][node.inputs[0]].get_shape()]
    return [cls.make_tensor_from_onnx_node(node, inputs=inputs, **kwargs)]
