import onnx_darknet.darknet.darknet_ctypes as dn

# from onnx_darknet.common import get_data_format
# from onnx_darknet.common import get_perm_from_formats
# from onnx_darknet.common import supports_device
from .broadcast_mixin import BroadcastMixin
# from .pad_mixin import PadMixin


class ConvMixin(BroadcastMixin):

    @classmethod
    def conv(cls, node, input_dict, transpose=False):
        """ Convolution method for both conv and transposed conv
        For transposed conv,
          Attr pads is not used for input, but declares how much output is padded.
          Here, output means output from transposed conv which already pad output_padding if set.
          So the pseudo explanation for output should be:
            output = conv_transpose_output + output_padding - pads
          And conv_transpose_output shape should be:
            conv_transpose_output_shape[i] = strides[i] * (input_shape[i] - 1) + kernel_shape[i]
        """
        # x is either previous layer or data tensor since Darknet layers do not
        # provide placeholder tensor objects
        x = input_dict[node.inputs[0]]
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

        # TODO(minhoolee): Figure out how to load weights and bias
        # (numpy arrays) to Darknet layer through custom load function after
        # creating the layer
        # Shape (M x C/group x KH x KW)
        weights = input_dict[node.inputs[1]]
        weights_spatial_shape = list(weights.shape)[2:]
        if "kernel_shape" in node.attrs.keys():
            kernel_shape = node.attrs["kernel_shape"]
            assert weights_spatial_shape == kernel_shape, (
                "kernel_shape attr of convolution does not match the actual "
                "weight passed to this operation, "
                "attr {}, actual {}").format(kernel_shape, weights.shape)

        nb_filters = weights.shape[1]
        k_size = weights.shape[2]
        dilations = node.attrs.get("dilations", [1] * spatial_size)
        # TODO(minhoolee): Figure out if and why Darknet is computing groups in
        # seequence and not in parallel
        group = node.attrs.get("group", 1)
        # TODO(minhoolee): Update Darknet im2col to separate padding and
        # support dilations like caffe2
        strides = node.attrs.get("strides", [1] * spatial_size)
        pads = node.attrs.get("pads", [0, 0] * spatial_size)

        if all(dim_size != k_size for dim_size in weights_spatial_shape):
            raise NotImplementedError(
                "Non-square convolutional kernel {} is not implemented in "
                "Darknet").format(weights_spatial_shape)

        if all(d != 1 for d in dilations):
            raise NotImplementedError(
                "Dilated convolution {} is not implemented in "
                "Darknet").format(dilations)

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

        # Darknet defaults will either be modified by future nodes
        # (e.g. activation) or will not be used during inference (e.g. adam)
        dn_defaults = {'activation': dn.ACTIVATION.RELU,
                       'batch_normalize': 0,
                       'binary': 0,
                       'xnor': 0,
                       'adam': 0}

        if transpose:
            layer = dn.make_deconvolutional_layer(N, H, W, C,
                                                  nb_filters, k_size, stride, pad,
                                                  dn_defaults['activation'],
                                                  dn_defaults['batch_normalize'],
                                                  dn_defaults['adam'])
        else:
            layer = dn.make_convolutional_layer(N, H, W, C,
                                                nb_filters, group, k_size, stride, pad,
                                                dn_defaults['activation'],
                                                dn_defaults['batch_normalize'],
                                                dn_defaults['binary'],
                                                dn_defaults['xnor'],
                                                dn_defaults['adam'])

        return [layer]
