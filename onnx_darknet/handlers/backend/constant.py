from onnx import numpy_helper
# import tensorflow as tf

from onnx_darknet.handlers.backend_handler import BackendHandler
from onnx_darknet.handlers.handler import onnx_op
from onnx_darknet.handlers.handler import darknet_func
from onnx_darknet.common import data_type


@onnx_op("Constant")
@darknet_func(tf.constant)
class Constant(BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    attr_value = node.attrs["value"]
    dtype = data_type.onnx2tf(attr_value.data_type)
    value = numpy_helper.to_array(attr_value)
    return [
        cls.make_tensor_from_onnx_node(
            node, inputs=[value], attrs={"dtype": dtype})
    ]
