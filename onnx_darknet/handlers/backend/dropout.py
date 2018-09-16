import onnx_darknet.darknet.darknet_ctypes as dn

from onnx_darknet.handlers.backend_handler import BackendHandler
from onnx_darknet.handlers.handler import onnx_op
from onnx_darknet.handlers.handler import darknet_func


@onnx_op("Dropout")
class Dropout(BackendHandler):

    @classmethod
    def _common(cls, node, **kwargs):
        x = kwargs["tensor_dict"][node.inputs[0]]
        if isinstance(x, dn.layer):
            N, nb_inputs = x.batch, x.outputs
        else:
            spatial_size = len(x.shape) - 2
            if spatial_size != 2:
                raise NotImplementedError(
                    "Dropout for {}d is not implemented "
                    "in Darknet").format(spatial_size)
            N, C, H, W = x.shape
            nb_inputs = C * H * W

        if cls.SINCE_VERSION <= 6 & node.attrs.get("is_test", 0):
            return np.empty(shape=[N, nb_inputs])

        ratio = node.attrs.get("ratio", 0.5)

        layer = dn.make_dropout_layer(N, nb_inputs, ratio)

        return [layer]

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_6(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_7(cls, node, **kwargs):
        return cls._common(node, **kwargs)
