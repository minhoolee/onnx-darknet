import onnx_darknet.darknet.darknet_ctypes as dn

from onnx_darknet.handlers.backend_handler import BackendHandler
from onnx_darknet.handlers.handler import onnx_op
from .pool_mixin import PoolMixin


@onnx_op("MaxPool")
class MaxPool(PoolMixin, BackendHandler):

    @classmethod
    def _common(cls, node, **kwargs):
        x = kwargs["tensor_dict"][node.inputs[0]]

        if isinstance(x, dn.layer):
            N, C, H, W = x.batch, x.out_c, x.out_h, x.out_w
            spatial_size = 2
        else:
            spatial_size = len(x.shape) - 2
            if spatial_size != 2:
                raise NotImplementedError(
                    "Convolution for {}d is not implemented "
                    "in Darknet").format(spatial_size)
            N, C, H, W = x.shape

        kernel_shape = node.attrs.get("kernel_shape", [1] * spatial_size)
        strides = node.attrs.get("strides", [1] * spatial_size)
        pads = node.attrs.get("pads", [0, 0] * spatial_size)

        kernel_size = kernel_shape[0]
        if all(dim_size != kernel_size for dim_size in kernel_shape):
            raise NotImplementedError(
                "Non-square max pool kernel {} is not implemented in "
                "Darknet").format(weights_spatial_shape)

        stride = strides[0]
        if all(s != stride for s in strides):
            raise NotImplementedError(
                "Unequal strides {} in different dimensions is not "
                "implemented in Darknet").format(strides.shape)

        pad = pads[0]
        if all(p != pad for p in pads):
            raise NotImplementedError(
                "Unequal pads {} for beginning and end of dimensions is not "
                "implemented in Darknet").format(pads.shape)

        # For some reason, Darknet max pool layer takes total pad and pads
        # input with total_pad / 2. This is different from Darknet conv layer
        layer = dn.make_maxpool_layer(N, H, W, C, kernel_size, stride, pad * 2)

        return [layer]

    @classmethod
    def version_1(cls, node, **kwargs):
        return cls._common(node, **kwargs)

    @classmethod
    def version_8(cls, node, **kwargs):
        return cls._common(node, **kwargs)
