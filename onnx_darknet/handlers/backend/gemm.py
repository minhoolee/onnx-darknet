import onnx_darknet.darknet.darknet_ctypes as dn

from onnx_darknet.handlers.backend_handler import BackendHandler
from onnx_darknet.handlers.handler import onnx_op


@onnx_op("Gemm")
class Gemm(BackendHandler):

    @classmethod
    def _common(cls, node, **kwargs):
        x = kwargs["tensor_dict"][node.inputs[0]]
        weights = kwargs["tensor_dict"][node.inputs[1]]

        if isinstance(x, dn.layer):
            print("Using layer {}".format(node.inputs[0]))
            M = x.batch
            K = x.outputs
        else:
            print("Using tensor {} ({})".format(node.inputs[0], x.shape))
            spatial_size = len(x.shape) - 2
            if spatial_size != 2:
                raise NotImplementedError(
                    "Connected layer for {}d is not implemented "
                    "in Darknet").format(spatial_size)
            M, C, H, W = x.shape
            K = C * H * W

        K_ = weights.shape[0]
        N = weights.shape[1]

        if node.attrs.get("transA", 0):
            M, K = K, M
        if node.attrs.get("transB", 0):
            K_, N = N, K_

        # TODO(minhoolee): Revert this; temporary fix for skipping max pool
        # layer's dimensionality reduction

        # assert K == K_, (
        #     "X.cols != Y.rows. X ({}, {}) and Y ({}, {}) cannot be multiplied "
        #     "with dot product").format(M, K, K_, N)
        K = K_

        alpha = node.attrs.get("alpha", 1.0)
        beta = node.attrs.get("beta", 1.0)

        if alpha != 1.0:
            raise NotImplementedError(
                "Alpha scaling factor for fully connected layer is not "
                "implemented in Darknet").format(weights_spatial_shape)

        if beta != 1.0:
            raise NotImplementedError(
                "Alpha scaling factor for fully connected layer is not "
                "implemented in Darknet").format(weights_spatial_shape)

        dn_defaults = {'activation': dn.ACTIVATION.LOGISTIC,
                       'batch_normalize': 0,
                       'adam': 0}

        layer = dn.make_connected_layer(M, K, N,
                                        dn_defaults['activation'],
                                        dn_defaults['batch_normalize'],
                                        dn_defaults['adam'])

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
