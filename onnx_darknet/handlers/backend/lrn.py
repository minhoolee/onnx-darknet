import copy

# import tensorflow as tf
import numpy as np

from onnx_darknet.handlers.backend_handler import BackendHandler
from onnx_darknet.handlers.handler import onnx_op
from onnx_darknet.handlers.handler import darknet_func


@onnx_op("LRN")
@darknet_func(tf.nn.lrn)
class LRN(BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    attrs = copy.deepcopy(node.attrs)
    alpha = attrs.get("alpha", 1e-4)
    attrs.setdefault("beta", 0.75)
    size = attrs["size"]
    attrs["alpha"] = alpha / size
    attrs["depth_radius"] = np.floor([(size - 1) / 2.])[0]
    # TODO: LRN in tf accepts radius
    # but in ONNX/Caffe accepts diameter.
    # This could be a problem.
    return [
        cls.make_tensor_from_onnx_node(
            node, attrs=attrs, c_last_only=True, **kwargs)
    ]
