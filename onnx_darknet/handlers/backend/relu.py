import onnx_darknet.darknet.darknet_ctypes as dn

from onnx_darknet.handlers.backend_handler import BackendHandler
from onnx_darknet.handlers.handler import onnx_op
from onnx_darknet.handlers.handler import darknet_func


@onnx_op("Relu")
class Relu(BackendHandler):

    @classmethod
    def _common(cls, node, **kwargs):
        layer = kwargs['tensor_dict'][node.inputs[0]]
        layer.activation = dn.ACTIVATION.RELU
        # TODO(minhoolee): Figure out better return value for non-layers
        return None

    @classmethod
    def version_1(cls, node, **kwargs):
        return [cls._common(node, **kwargs)]

    @classmethod
    def version_6(cls, node, **kwargs):
        return [cls._common(node, **kwargs)]
