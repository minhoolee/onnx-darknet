import numpy as np

import onnx_darknet.darknet.darknet_ctypes as dn

from onnx_darknet.handlers.backend_handler import BackendHandler
from onnx_darknet.handlers.handler import onnx_op
from onnx_darknet.handlers.handler import darknet_func


@onnx_op("Relu")
class Relu(BackendHandler):

    @classmethod
    def _common(cls, node, **kwargs):
        x = kwargs['tensor_dict'][node.inputs[0]]

        assert isinstance(x, dn.layer), (
            "Darknet activation {} must be applied after Darknet layer "
            "instead of {}").format(cls.ONNX_OP, x)

        x.activation = dn.ACTIVATION.RELU

        # TODO(minhoolee): Figure out better return value
        return [np.empty(shape=[x.batch, x.out_c, x.out_h, x.out_w])]

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_6(cls, node, **kwargs):
        return cls._common(node, **kwargs)
