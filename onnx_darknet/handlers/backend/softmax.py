import onnx_darknet.darknet.darknet_ctypes as dn

from onnx_darknet.handlers.backend_handler import BackendHandler
from onnx_darknet.handlers.handler import onnx_op
from onnx_darknet.handlers.handler import darknet_func


@onnx_op("Softmax")
class Softmax(BackendHandler):

    @classmethod
    def version_1(cls, node, **kwargs):
        x = kwargs["tensor_dict"][node.inputs[0]]

        if isinstance(x, dn.layer):
            N, nb_inputs = x.batch, x.outputs
        else:
            spatial_size = len(x.shape) - 2
            if spatial_size != 2:
                raise NotImplementedError(
                    "Softmax for {}d is not implemented "
                    "in Darknet").format(spatial_size)
            N, C, H, W = x.shape
            nb_inputs = C * H * W

        # ONNX does not support grouped softmax
        dn_defaults = {'groups': 1}

        layer = dn.make_softmax_layer(N, nb_inputs, dn_defaults['groups'])

        return [layer]
