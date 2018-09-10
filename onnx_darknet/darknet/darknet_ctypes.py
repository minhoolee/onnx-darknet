# -*- coding: utf-8 -*-
"""Temporary frontend to Darknet library

To run this, you will need to have libdarknet.so built from
Darknet and moved to onnx_darknet/darknet/libdarknet.so
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import ctypes
import math
import random

import onnx_darknet.darknet.cudnn_ctypes as cudnn

# def sample(probs):
#     s = sum(probs)
#     probs = [a/s for a in probs]
#     r = random.uniform(0, 1)
#     for i in range(len(probs)):
#         r = r - probs[i]
#         if r <= 0:
#             return i
#     return len(probs)-1
#
# def c_array(ctype, values):
#     arr = (ctype*len(values))()
#     arr[:] = values
#     return arr
#
# def np_array_to_image(arr):
#     import numpy as np
#     if type(arr) is not np.array
#     arr = arr.transpose(2,0,1)
#     c = arr.shape[0]
#     h = arr.shape[1]
#     w = arr.shape[2]
#     arr = (arr/255.0).flatten()
#     data = c_array(ctypes.c_float, arr)
#     im = image(w,h,c,data)
#     return im

# if local wordsize is same as target, keep ctypes pointer function.
if ctypes.sizeof(ctypes.c_void_p) == 8:
    POINTER_T = ctypes.POINTER
else:
    # required to access _ctypes
    import _ctypes
    # Emulate a pointer class using the approriate c_int32/c_int64 type
    # The new class should have :
    # ['__module__', 'from_param', '_type_', '__dict__', '__weakref__', '__doc__']
    # but the class should be submitted to a unique instance for each base type
    # to that if A == B, POINTER_T(A) == POINTER_T(B)
    ctypes._pointer_t_type_cache = {}
    def POINTER_T(pointee):
        # a pointer should have the same length as LONG
        fake_ptr_base_type = ctypes.c_uint64
        # specific case for c_void_p
        if pointee is None: # VOID pointer type. c_void_p.
            pointee = type(None) # ctypes.c_void_p # ctypes.c_ulong
            clsname = 'c_void'
        else:
            clsname = pointee.__name__
        if clsname in ctypes._pointer_t_type_cache:
            return ctypes._pointer_t_type_cache[clsname]
        # make template
        class _T(_ctypes._SimpleCData,):
            _type_ = 'L'
            _subtype_ = pointee
            def _sub_addr_(self):
                return self.value
            def __repr__(self):
                return '%s(%d)'%(clsname, self.value)
            def contents(self):
                raise TypeError('This is not a ctypes pointer.')
            def __init__(self, **args):
                raise TypeError('This is not a ctypes pointer. It is not instanciable.')
        _class = type('LP_%d_%s'%(8, clsname), (_T,),{})
        ctypes._pointer_t_type_cache[clsname] = _class
        return _class

ctypes.c_int128 = ctypes.c_ubyte*16
ctypes.c_uint128 = ctypes.c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 16:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*16

gpu_index = None # Externed variable ctypes.c_int32

class EnumerationType(type(ctypes.c_uint)):
    def __new__(metacls, name, bases, dict):
        if not "_members_" in dict:
            _members_ = {}
            for key, value in dict.items():
                if not key.startswith("_"):
                    _members_[key] = value

            dict["_members_"] = _members_
        else:
            _members_ = dict["_members_"]

        dict["_reverse_map_"] = { v: k for k, v in _members_.items() }
        cls = type(ctypes.c_uint).__new__(metacls, name, bases, dict)
        for key,value in cls._members_.items():
            globals()[key] = value
        return cls

    def __repr__(self):
        return "<Enumeration %s>" % self.__name__

class CEnumeration(ctypes.c_uint, metaclass = EnumerationType):
    _members_     = {}

    def __repr__(self):
        value = self.value
        return "<%s.%s: %d>" % (
            self.__class__.__name__,
            self._reverse_map_.get(value, '(unknown)'),
            value
        )

    def __eq__(self, other):
        if isinstance(other, int):
            return self.value == other

        return type(self) == type(other) and self.value == other.value

# Enums
class ACTIVATION(CEnumeration):
    LOGISTIC = 0
    RELU = 1
    RELIE = 2
    LINEAR = 3
    RAMP = 4
    TANH = 5
    PLSE = 6
    LEAKY = 7
    ELU = 8
    LOGGY = 9
    STAIR = 10
    HARDTAN = 11
    LHTAN = 12

class BINARY_ACTIVATION(CEnumeration):
    MULT = 0
    ADD = 1
    SUB = 2
    DIV = 3

class LAYER_TYPE(CEnumeration):
    CONVOLUTIONAL = 0
    DECONVOLUTIONAL = 1
    CONNECTED = 2
    MAXPOOL = 3
    SOFTMAX = 4
    DETECTION = 5
    DROPOUT = 6
    CROP = 7
    ROUTE = 8
    COST = 9
    NORMALIZATION = 10
    AVGPOOL = 11
    LOCAL = 12
    SHORTCUT = 13
    ACTIVE = 14
    RNN = 15
    GRU = 16
    LSTM = 17
    CRNN = 18
    BATCHNORM = 19
    NETWORK = 20
    XNOR = 21
    REGION = 22
    YOLO = 23
    ISEG = 24
    REORG = 25
    UPSAMPLE = 26
    LOGXENT = 27
    L2NORM = 28
    BLANK = 29

class COST_TYPE(CEnumeration):
    SSE = 0
    MASKED = 1
    L1 = 2
    SEG = 3
    SMOOTH = 4
    WGAN = 5

class learning_rate_policy(CEnumeration):
    CONSTANT = 0
    STEP = 1
    EXP = 2
    POLY = 3
    STEPS = 4
    SIG = 5
    RANDOM = 6

class data_type(CEnumeration):
    CLASSIFICATION_DATA = 0
    DETECTION_DATA = 1
    CAPTCHA_DATA = 2
    REGION_DATA = 3
    IMAGE_DATA = 4
    COMPARE_DATA = 5
    WRITING_DATA = 6
    SWAG_DATA = 7
    TAG_DATA = 8
    OLD_CLASSIFICATION_DATA = 9
    STUDY_DATA = 10
    DET_DATA = 11
    SUPER_DATA = 12
    LETTERBOX_DATA = 13
    REGRESSION_DATA = 14
    SEGMENTATION_DATA = 15
    INSTANCE_DATA = 16
    ISEG_DATA = 17

# Structs
class metadata(ctypes.Structure):
    _pack_ = True # source:False
    _fields_ = [
    ('classes', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('names', POINTER_T(ctypes.c_char_p)),
     ]

class tree(ctypes.Structure):
    _pack_ = True # source:False
    _fields_ = [
    ('leaf', POINTER_T(ctypes.c_int32)),
    ('n', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('parent', POINTER_T(ctypes.c_int32)),
    ('child', POINTER_T(ctypes.c_int32)),
    ('group', POINTER_T(ctypes.c_int32)),
    ('name', POINTER_T(ctypes.c_char_p)),
    ('groups', ctypes.c_int32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('group_size', POINTER_T(ctypes.c_int32)),
    ('group_offset', POINTER_T(ctypes.c_int32)),
     ]

class update_args(ctypes.Structure):
    _pack_ = True # source:False
    _fields_ = [
    ('batch', ctypes.c_int32),
    ('learning_rate', ctypes.c_float),
    ('momentum', ctypes.c_float),
    ('decay', ctypes.c_float),
    ('adam', ctypes.c_int32),
    ('B1', ctypes.c_float),
    ('B2', ctypes.c_float),
    ('eps', ctypes.c_float),
    ('t', ctypes.c_int32),
     ]

class layer(ctypes.Structure):
    pass

class network(ctypes.Structure):
    pass

layer._pack_ = True # source:False
layer._fields_ = [
    ('type', LAYER_TYPE),
    ('activation', ACTIVATION),
    ('cost_type', COST_TYPE),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('forward', POINTER_T(ctypes.CFUNCTYPE(None, layer, network))),
    ('backward', POINTER_T(ctypes.CFUNCTYPE(None, layer, network))),
    ('update', POINTER_T(ctypes.CFUNCTYPE(None, layer, update_args))),
    ('forward_gpu', POINTER_T(ctypes.CFUNCTYPE(None, layer, network))),
    ('backward_gpu', POINTER_T(ctypes.CFUNCTYPE(None, layer, network))),
    ('update_gpu', POINTER_T(ctypes.CFUNCTYPE(None, layer, update_args))),
    ('batch_normalize', ctypes.c_int32),
    ('shortcut', ctypes.c_int32),
    ('batch', ctypes.c_int32),
    ('forced', ctypes.c_int32),
    ('flipped', ctypes.c_int32),
    ('inputs', ctypes.c_int32),
    ('outputs', ctypes.c_int32),
    ('nweights', ctypes.c_int32),
    ('nbiases', ctypes.c_int32),
    ('extra', ctypes.c_int32),
    ('truths', ctypes.c_int32),
    ('h', ctypes.c_int32),
    ('w', ctypes.c_int32),
    ('c', ctypes.c_int32),
    ('out_h', ctypes.c_int32),
    ('out_w', ctypes.c_int32),
    ('out_c', ctypes.c_int32),
    ('n', ctypes.c_int32),
    ('max_boxes', ctypes.c_int32),
    ('groups', ctypes.c_int32),
    ('size', ctypes.c_int32),
    ('side', ctypes.c_int32),
    ('stride', ctypes.c_int32),
    ('reverse', ctypes.c_int32),
    ('flatten', ctypes.c_int32),
    ('spatial', ctypes.c_int32),
    ('pad', ctypes.c_int32),
    ('sqrt', ctypes.c_int32),
    ('flip', ctypes.c_int32),
    ('index', ctypes.c_int32),
    ('binary', ctypes.c_int32),
    ('xnor', ctypes.c_int32),
    ('steps', ctypes.c_int32),
    ('hidden', ctypes.c_int32),
    ('truth', ctypes.c_int32),
    ('smooth', ctypes.c_float),
    ('dot', ctypes.c_float),
    ('angle', ctypes.c_float),
    ('jitter', ctypes.c_float),
    ('saturation', ctypes.c_float),
    ('exposure', ctypes.c_float),
    ('shift', ctypes.c_float),
    ('ratio', ctypes.c_float),
    ('learning_rate_scale', ctypes.c_float),
    ('clip', ctypes.c_float),
    ('noloss', ctypes.c_int32),
    ('softmax', ctypes.c_int32),
    ('classes', ctypes.c_int32),
    ('coords', ctypes.c_int32),
    ('background', ctypes.c_int32),
    ('rescore', ctypes.c_int32),
    ('objectness', ctypes.c_int32),
    ('joint', ctypes.c_int32),
    ('noadjust', ctypes.c_int32),
    ('reorg', ctypes.c_int32),
    ('log', ctypes.c_int32),
    ('tanh', ctypes.c_int32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('mask', POINTER_T(ctypes.c_int32)),
    ('total', ctypes.c_int32),
    ('alpha', ctypes.c_float),
    ('beta', ctypes.c_float),
    ('kappa', ctypes.c_float),
    ('coord_scale', ctypes.c_float),
    ('object_scale', ctypes.c_float),
    ('noobject_scale', ctypes.c_float),
    ('mask_scale', ctypes.c_float),
    ('class_scale', ctypes.c_float),
    ('bias_match', ctypes.c_int32),
    ('random', ctypes.c_int32),
    ('ignore_thresh', ctypes.c_float),
    ('truth_thresh', ctypes.c_float),
    ('thresh', ctypes.c_float),
    ('focus', ctypes.c_float),
    ('classfix', ctypes.c_int32),
    ('absolute', ctypes.c_int32),
    ('onlyforward', ctypes.c_int32),
    ('stopbackward', ctypes.c_int32),
    ('dontload', ctypes.c_int32),
    ('dontsave', ctypes.c_int32),
    ('dontloadscales', ctypes.c_int32),
    ('numload', ctypes.c_int32),
    ('temperature', ctypes.c_float),
    ('probability', ctypes.c_float),
    ('scale', ctypes.c_float),
    ('cweights', ctypes.c_char_p),
    ('indexes', POINTER_T(ctypes.c_int32)),
    ('input_layers', POINTER_T(ctypes.c_int32)),
    ('input_sizes', POINTER_T(ctypes.c_int32)),
    ('map', POINTER_T(ctypes.c_int32)),
    ('counts', POINTER_T(ctypes.c_int32)),
    ('sums', POINTER_T(POINTER_T(ctypes.c_float))),
    ('rand', POINTER_T(ctypes.c_float)),
    ('cost', POINTER_T(ctypes.c_float)),
    ('state', POINTER_T(ctypes.c_float)),
    ('prev_state', POINTER_T(ctypes.c_float)),
    ('forgot_state', POINTER_T(ctypes.c_float)),
    ('forgot_delta', POINTER_T(ctypes.c_float)),
    ('state_delta', POINTER_T(ctypes.c_float)),
    ('combine_cpu', POINTER_T(ctypes.c_float)),
    ('combine_delta_cpu', POINTER_T(ctypes.c_float)),
    ('concat', POINTER_T(ctypes.c_float)),
    ('concat_delta', POINTER_T(ctypes.c_float)),
    ('binary_weights', POINTER_T(ctypes.c_float)),
    ('biases', POINTER_T(ctypes.c_float)),
    ('bias_updates', POINTER_T(ctypes.c_float)),
    ('scales', POINTER_T(ctypes.c_float)),
    ('scale_updates', POINTER_T(ctypes.c_float)),
    ('weights', POINTER_T(ctypes.c_float)),
    ('weight_updates', POINTER_T(ctypes.c_float)),
    ('delta', POINTER_T(ctypes.c_float)),
    ('output', POINTER_T(ctypes.c_float)),
    ('loss', POINTER_T(ctypes.c_float)),
    ('squared', POINTER_T(ctypes.c_float)),
    ('norms', POINTER_T(ctypes.c_float)),
    ('spatial_mean', POINTER_T(ctypes.c_float)),
    ('mean', POINTER_T(ctypes.c_float)),
    ('variance', POINTER_T(ctypes.c_float)),
    ('mean_delta', POINTER_T(ctypes.c_float)),
    ('variance_delta', POINTER_T(ctypes.c_float)),
    ('rolling_mean', POINTER_T(ctypes.c_float)),
    ('rolling_variance', POINTER_T(ctypes.c_float)),
    ('x', POINTER_T(ctypes.c_float)),
    ('x_norm', POINTER_T(ctypes.c_float)),
    ('m', POINTER_T(ctypes.c_float)),
    ('v', POINTER_T(ctypes.c_float)),
    ('bias_m', POINTER_T(ctypes.c_float)),
    ('bias_v', POINTER_T(ctypes.c_float)),
    ('scale_m', POINTER_T(ctypes.c_float)),
    ('scale_v', POINTER_T(ctypes.c_float)),
    ('z_cpu', POINTER_T(ctypes.c_float)),
    ('r_cpu', POINTER_T(ctypes.c_float)),
    ('h_cpu', POINTER_T(ctypes.c_float)),
    ('prev_state_cpu', POINTER_T(ctypes.c_float)),
    ('temp_cpu', POINTER_T(ctypes.c_float)),
    ('temp2_cpu', POINTER_T(ctypes.c_float)),
    ('temp3_cpu', POINTER_T(ctypes.c_float)),
    ('dh_cpu', POINTER_T(ctypes.c_float)),
    ('hh_cpu', POINTER_T(ctypes.c_float)),
    ('prev_cell_cpu', POINTER_T(ctypes.c_float)),
    ('cell_cpu', POINTER_T(ctypes.c_float)),
    ('f_cpu', POINTER_T(ctypes.c_float)),
    ('i_cpu', POINTER_T(ctypes.c_float)),
    ('g_cpu', POINTER_T(ctypes.c_float)),
    ('o_cpu', POINTER_T(ctypes.c_float)),
    ('c_cpu', POINTER_T(ctypes.c_float)),
    ('dc_cpu', POINTER_T(ctypes.c_float)),
    ('binary_input', POINTER_T(ctypes.c_float)),
    ('input_layer', POINTER_T(layer)),
    ('self_layer', POINTER_T(layer)),
    ('output_layer', POINTER_T(layer)),
    ('reset_layer', POINTER_T(layer)),
    ('update_layer', POINTER_T(layer)),
    ('state_layer', POINTER_T(layer)),
    ('input_gate_layer', POINTER_T(layer)),
    ('state_gate_layer', POINTER_T(layer)),
    ('input_save_layer', POINTER_T(layer)),
    ('state_save_layer', POINTER_T(layer)),
    ('input_state_layer', POINTER_T(layer)),
    ('state_state_layer', POINTER_T(layer)),
    ('input_z_layer', POINTER_T(layer)),
    ('state_z_layer', POINTER_T(layer)),
    ('input_r_layer', POINTER_T(layer)),
    ('state_r_layer', POINTER_T(layer)),
    ('input_h_layer', POINTER_T(layer)),
    ('state_h_layer', POINTER_T(layer)),
    ('wz', POINTER_T(layer)),
    ('uz', POINTER_T(layer)),
    ('wr', POINTER_T(layer)),
    ('ur', POINTER_T(layer)),
    ('wh', POINTER_T(layer)),
    ('uh', POINTER_T(layer)),
    ('uo', POINTER_T(layer)),
    ('wo', POINTER_T(layer)),
    ('uf', POINTER_T(layer)),
    ('wf', POINTER_T(layer)),
    ('ui', POINTER_T(layer)),
    ('wi', POINTER_T(layer)),
    ('ug', POINTER_T(layer)),
    ('wg', POINTER_T(layer)),
    ('softmax_tree', POINTER_T(tree)),
    ('workspace_size', ctypes.c_uint64),
    ('indexes_gpu', POINTER_T(ctypes.c_int32)),
    ('z_gpu', POINTER_T(ctypes.c_float)),
    ('r_gpu', POINTER_T(ctypes.c_float)),
    ('h_gpu', POINTER_T(ctypes.c_float)),
    ('temp_gpu', POINTER_T(ctypes.c_float)),
    ('temp2_gpu', POINTER_T(ctypes.c_float)),
    ('temp3_gpu', POINTER_T(ctypes.c_float)),
    ('dh_gpu', POINTER_T(ctypes.c_float)),
    ('hh_gpu', POINTER_T(ctypes.c_float)),
    ('prev_cell_gpu', POINTER_T(ctypes.c_float)),
    ('cell_gpu', POINTER_T(ctypes.c_float)),
    ('f_gpu', POINTER_T(ctypes.c_float)),
    ('i_gpu', POINTER_T(ctypes.c_float)),
    ('g_gpu', POINTER_T(ctypes.c_float)),
    ('o_gpu', POINTER_T(ctypes.c_float)),
    ('c_gpu', POINTER_T(ctypes.c_float)),
    ('dc_gpu', POINTER_T(ctypes.c_float)),
    ('m_gpu', POINTER_T(ctypes.c_float)),
    ('v_gpu', POINTER_T(ctypes.c_float)),
    ('bias_m_gpu', POINTER_T(ctypes.c_float)),
    ('scale_m_gpu', POINTER_T(ctypes.c_float)),
    ('bias_v_gpu', POINTER_T(ctypes.c_float)),
    ('scale_v_gpu', POINTER_T(ctypes.c_float)),
    ('combine_gpu', POINTER_T(ctypes.c_float)),
    ('combine_delta_gpu', POINTER_T(ctypes.c_float)),
    ('prev_state_gpu', POINTER_T(ctypes.c_float)),
    ('forgot_state_gpu', POINTER_T(ctypes.c_float)),
    ('forgot_delta_gpu', POINTER_T(ctypes.c_float)),
    ('state_gpu', POINTER_T(ctypes.c_float)),
    ('state_delta_gpu', POINTER_T(ctypes.c_float)),
    ('gate_gpu', POINTER_T(ctypes.c_float)),
    ('gate_delta_gpu', POINTER_T(ctypes.c_float)),
    ('save_gpu', POINTER_T(ctypes.c_float)),
    ('save_delta_gpu', POINTER_T(ctypes.c_float)),
    ('concat_gpu', POINTER_T(ctypes.c_float)),
    ('concat_delta_gpu', POINTER_T(ctypes.c_float)),
    ('binary_input_gpu', POINTER_T(ctypes.c_float)),
    ('binary_weights_gpu', POINTER_T(ctypes.c_float)),
    ('mean_gpu', POINTER_T(ctypes.c_float)),
    ('variance_gpu', POINTER_T(ctypes.c_float)),
    ('rolling_mean_gpu', POINTER_T(ctypes.c_float)),
    ('rolling_variance_gpu', POINTER_T(ctypes.c_float)),
    ('variance_delta_gpu', POINTER_T(ctypes.c_float)),
    ('mean_delta_gpu', POINTER_T(ctypes.c_float)),
    ('x_gpu', POINTER_T(ctypes.c_float)),
    ('x_norm_gpu', POINTER_T(ctypes.c_float)),
    ('weights_gpu', POINTER_T(ctypes.c_float)),
    ('weight_updates_gpu', POINTER_T(ctypes.c_float)),
    ('weight_change_gpu', POINTER_T(ctypes.c_float)),
    ('biases_gpu', POINTER_T(ctypes.c_float)),
    ('bias_updates_gpu', POINTER_T(ctypes.c_float)),
    ('bias_change_gpu', POINTER_T(ctypes.c_float)),
    ('scales_gpu', POINTER_T(ctypes.c_float)),
    ('scale_updates_gpu', POINTER_T(ctypes.c_float)),
    ('scale_change_gpu', POINTER_T(ctypes.c_float)),
    ('output_gpu', POINTER_T(ctypes.c_float)),
    ('loss_gpu', POINTER_T(ctypes.c_float)),
    ('delta_gpu', POINTER_T(ctypes.c_float)),
    ('rand_gpu', POINTER_T(ctypes.c_float)),
    ('squared_gpu', POINTER_T(ctypes.c_float)),
    ('norms_gpu', POINTER_T(ctypes.c_float)),
    ('srcTensorDesc', POINTER_T(cudnn.struct_cudnnTensorStruct)),
    ('dstTensorDesc', POINTER_T(cudnn.struct_cudnnTensorStruct)),
    ('dsrcTensorDesc', POINTER_T(cudnn.struct_cudnnTensorStruct)),
    ('ddstTensorDesc', POINTER_T(cudnn.struct_cudnnTensorStruct)),
    ('normTensorDesc', POINTER_T(cudnn.struct_cudnnTensorStruct)),
    ('weightDesc', POINTER_T(cudnn.struct_cudnnFilterStruct)),
    ('dweightDesc', POINTER_T(cudnn.struct_cudnnFilterStruct)),
    ('convDesc', POINTER_T(cudnn.struct_cudnnConvolutionStruct)),
    ('fw_algo', cudnn.c__EA_cudnnConvolutionFwdAlgo_t),
    ('bd_algo', cudnn.c__EA_cudnnConvolutionBwdDataAlgo_t),
    ('bf_algo', cudnn.c__EA_cudnnConvolutionBwdFilterAlgo_t),
    ('PADDING_2', ctypes.c_ubyte * 4),
]

network._pack_ = True # source:False
network._fields_ = [
    ('n', ctypes.c_int32),
    ('batch', ctypes.c_int32),
    ('seen', POINTER_T(ctypes.c_uint64)),
    ('t', POINTER_T(ctypes.c_int32)),
    ('epoch', ctypes.c_float),
    ('subdivisions', ctypes.c_int32),
    ('layers', POINTER_T(layer)),
    ('output', POINTER_T(ctypes.c_float)),
    ('policy', learning_rate_policy),
    ('learning_rate', ctypes.c_float),
    ('momentum', ctypes.c_float),
    ('decay', ctypes.c_float),
    ('gamma', ctypes.c_float),
    ('scale', ctypes.c_float),
    ('power', ctypes.c_float),
    ('time_steps', ctypes.c_int32),
    ('step', ctypes.c_int32),
    ('max_batches', ctypes.c_int32),
    ('scales', POINTER_T(ctypes.c_float)),
    ('steps', POINTER_T(ctypes.c_int32)),
    ('num_steps', ctypes.c_int32),
    ('burn_in', ctypes.c_int32),
    ('adam', ctypes.c_int32),
    ('B1', ctypes.c_float),
    ('B2', ctypes.c_float),
    ('eps', ctypes.c_float),
    ('inputs', ctypes.c_int32),
    ('outputs', ctypes.c_int32),
    ('truths', ctypes.c_int32),
    ('notruth', ctypes.c_int32),
    ('h', ctypes.c_int32),
    ('w', ctypes.c_int32),
    ('c', ctypes.c_int32),
    ('max_crop', ctypes.c_int32),
    ('min_crop', ctypes.c_int32),
    ('max_ratio', ctypes.c_float),
    ('min_ratio', ctypes.c_float),
    ('center', ctypes.c_int32),
    ('angle', ctypes.c_float),
    ('aspect', ctypes.c_float),
    ('exposure', ctypes.c_float),
    ('saturation', ctypes.c_float),
    ('hue', ctypes.c_float),
    ('random', ctypes.c_int32),
    ('gpu_index', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('hierarchy', POINTER_T(tree)),
    ('input', POINTER_T(ctypes.c_float)),
    ('truth', POINTER_T(ctypes.c_float)),
    ('delta', POINTER_T(ctypes.c_float)),
    ('workspace', POINTER_T(ctypes.c_float)),
    ('train', ctypes.c_int32),
    ('index', ctypes.c_int32),
    ('cost', POINTER_T(ctypes.c_float)),
    ('clip', ctypes.c_float),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('input_gpu', POINTER_T(ctypes.c_float)),
    ('truth_gpu', POINTER_T(ctypes.c_float)),
    ('delta_gpu', POINTER_T(ctypes.c_float)),
    ('output_gpu', POINTER_T(ctypes.c_float)),
]

class augment_args(ctypes.Structure):
    _pack_ = True # source:False
    _fields_ = [
    ('w', ctypes.c_int32),
    ('h', ctypes.c_int32),
    ('scale', ctypes.c_float),
    ('rad', ctypes.c_float),
    ('dx', ctypes.c_float),
    ('dy', ctypes.c_float),
    ('aspect', ctypes.c_float),
     ]

class image(ctypes.Structure):
    _pack_ = True # source:False
    _fields_ = [
    ('w', ctypes.c_int32),
    ('h', ctypes.c_int32),
    ('c', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('data', POINTER_T(ctypes.c_float)),
     ]

class box(ctypes.Structure):
    _pack_ = True # source:False
    _fields_ = [
    ('x', ctypes.c_float),
    ('y', ctypes.c_float),
    ('w', ctypes.c_float),
    ('h', ctypes.c_float),
     ]

class detection(ctypes.Structure):
    _pack_ = True # source:False
    _fields_ = [
    ('bbox', box),
    ('classes', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('prob', POINTER_T(ctypes.c_float)),
    ('mask', POINTER_T(ctypes.c_float)),
    ('objectness', ctypes.c_float),
    ('sort_class', ctypes.c_int32),
     ]

class matrix(ctypes.Structure):
    _pack_ = True # source:False
    _fields_ = [
    ('rows', ctypes.c_int32),
    ('cols', ctypes.c_int32),
    ('vals', POINTER_T(POINTER_T(ctypes.c_float))),
     ]

class data(ctypes.Structure):
    _pack_ = True # source:False
    _fields_ = [
    ('w', ctypes.c_int32),
    ('h', ctypes.c_int32),
    ('X', matrix),
    ('y', matrix),
    ('shallow', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('num_boxes', POINTER_T(ctypes.c_int32)),
    ('boxes', POINTER_T(POINTER_T(box))),
     ]

class load_args(ctypes.Structure):
    _pack_ = True # source:False
    _fields_ = [
    ('threads', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('paths', POINTER_T(ctypes.c_char_p)),
    ('path', ctypes.c_char_p),
    ('n', ctypes.c_int32),
    ('m', ctypes.c_int32),
    ('labels', POINTER_T(ctypes.c_char_p)),
    ('h', ctypes.c_int32),
    ('w', ctypes.c_int32),
    ('out_w', ctypes.c_int32),
    ('out_h', ctypes.c_int32),
    ('nh', ctypes.c_int32),
    ('nw', ctypes.c_int32),
    ('num_boxes', ctypes.c_int32),
    ('min', ctypes.c_int32),
    ('max', ctypes.c_int32),
    ('size', ctypes.c_int32),
    ('classes', ctypes.c_int32),
    ('background', ctypes.c_int32),
    ('scale', ctypes.c_int32),
    ('center', ctypes.c_int32),
    ('coords', ctypes.c_int32),
    ('jitter', ctypes.c_float),
    ('angle', ctypes.c_float),
    ('aspect', ctypes.c_float),
    ('saturation', ctypes.c_float),
    ('exposure', ctypes.c_float),
    ('hue', ctypes.c_float),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('d', POINTER_T(data)),
    ('im', POINTER_T(image)),
    ('resized', POINTER_T(image)),
    ('type', data_type),
    ('PADDING_2', ctypes.c_ubyte * 4),
    ('hierarchy', POINTER_T(tree)),
     ]

class box_label(ctypes.Structure):
    _pack_ = True # source:False
    _fields_ = [
    ('id', ctypes.c_int32),
    ('x', ctypes.c_float),
    ('y', ctypes.c_float),
    ('w', ctypes.c_float),
    ('h', ctypes.c_float),
    ('left', ctypes.c_float),
    ('right', ctypes.c_float),
    ('top', ctypes.c_float),
    ('bottom', ctypes.c_float),
     ]

class node(ctypes.Structure):
    pass

node._pack_ = True # source:False
node._fields_ = [
    ('val', POINTER_T(None)),
    ('next', POINTER_T(node)),
    ('prev', POINTER_T(node)),
]

class list(ctypes.Structure):
    _pack_ = True # source:False
    _fields_ = [
    ('size', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('front', POINTER_T(node)),
    ('back', POINTER_T(node)),
     ]


darknet_c = ctypes.CDLL("onnx_darknet/darknet/libdarknet.so", ctypes.RTLD_GLOBAL)

network_width = darknet_c.network_width
network_width.argtypes = [ctypes.c_void_p]
network_width.restype = ctypes.c_int

network_height = darknet_c.network_height
network_height.argtypes = [ctypes.c_void_p]
network_height.restype = ctypes.c_int

network_predict = darknet_c.network_predict
network_predict.argtypes = [ctypes.c_void_p, POINTER_T(ctypes.c_float)]
network_predict.restype = POINTER_T(ctypes.c_float)

set_gpu = darknet_c.cuda_set_device
set_gpu.argtypes = [ctypes.c_int]

make_image = darknet_c.make_image
make_image.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
make_image.restype = image

get_network_boxes = darknet_c.get_network_boxes
get_network_boxes.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, POINTER_T(ctypes.c_int), ctypes.c_int, POINTER_T(ctypes.c_int)]
get_network_boxes.restype = POINTER_T(detection)

make_network_boxes = darknet_c.make_network_boxes
make_network_boxes.argtypes = [ctypes.c_void_p]
make_network_boxes.restype = POINTER_T(detection)

free_detections = darknet_c.free_detections
free_detections.argtypes = [POINTER_T(detection), ctypes.c_int]

free_ptrs = darknet_c.free_ptrs
free_ptrs.argtypes = [POINTER_T(ctypes.c_void_p), ctypes.c_int]

network_predict = darknet_c.network_predict
network_predict.argtypes = [ctypes.c_void_p, POINTER_T(ctypes.c_float)]

reset_rnn = darknet_c.reset_rnn
reset_rnn.argtypes = [ctypes.c_void_p]

do_nms_obj = darknet_c.do_nms_obj
do_nms_obj.argtypes = [POINTER_T(detection), ctypes.c_int, ctypes.c_int, ctypes.c_float]

do_nms_sort = darknet_c.do_nms_sort
do_nms_sort.argtypes = [POINTER_T(detection), ctypes.c_int, ctypes.c_int, ctypes.c_float]

free_image = darknet_c.free_image
free_image.argtypes = [image]

letterbox_image = darknet_c.letterbox_image
letterbox_image.argtypes = [image, ctypes.c_int, ctypes.c_int]
letterbox_image.restype = image

get_metadata = darknet_c.get_metadata
get_metadata.argtypes = [ctypes.c_char_p]
get_metadata.restype = metadata

load_image_color = darknet_c.load_image_color
load_image_color.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
load_image_color.restype = image

rgbgr_image = darknet_c.rgbgr_image
rgbgr_image.argtypes = [image]
get_layer_string = darknet_c.get_layer_string
get_layer_string.argtypes = [LAYER_TYPE]
get_layer_string.restype = ctypes.c_char_p

load_alphabet = darknet_c.load_alphabet
load_alphabet.restype = POINTER_T(POINTER_T(image))

draw_detections = darknet_c.draw_detections
draw_detections.argtypes = [image, POINTER_T(detection), ctypes.c_int, ctypes.c_float, POINTER_T(ctypes.c_char_p), POINTER_T(POINTER_T(image)), ctypes.c_int]

save_image = darknet_c.save_image
save_image.argtypes = [image, ctypes.c_char_p]

load_network = darknet_c.load_network
load_network.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
load_network.restype = POINTER_T(network)

parse_network_cfg = darknet_c.parse_network_cfg
parse_network_cfg.argtypes = [ctypes.c_char_p]
parse_network_cfg.restype = POINTER_T(network)

make_network = darknet_c.make_network
make_network.argtypes = [ctypes.c_int]
make_network.restype = POINTER_T(network)

make_connected_layer = darknet_c.make_connected_layer
make_connected_layer.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ACTIVATION, ctypes.c_int, ctypes.c_int]
make_connected_layer.restype = layer

make_convolutional_layer = darknet_c.make_convolutional_layer
make_convolutional_layer.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ACTIVATION, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
make_convolutional_layer.restype = layer

make_dropout_layer = darknet_c.make_dropout_layer
make_dropout_layer.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float]
make_dropout_layer.restype = layer

make_maxpool_layer = darknet_c.make_maxpool_layer
make_maxpool_layer.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
make_maxpool_layer.restype = layer

make_softmax_layer = darknet_c.make_softmax_layer
make_softmax_layer.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
make_softmax_layer.restype = layer

make_crop_layer = darknet_c.make_crop_layer
make_crop_layer.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ACTIVATION]
make_crop_layer.restype = layer

network_predict_image = darknet_c.network_predict_image
network_predict_image.argtypes = [ctypes.c_void_p, image]
network_predict_image.restype = POINTER_T(ctypes.c_float)

free_network = darknet_c.free_network
free_network.argtypes = [POINTER_T(network)]

print_network = darknet_c.print_network
print_network.argtypes = [POINTER_T(network)]

visualize_network = darknet_c.visualize_network
visualize_network.argtypes = [POINTER_T(network)]

def classify(net, meta, im):
    out = network_predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i].decode('utf-8'), out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, out_img, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image_color(image, 0, 0)

    # Get filename without extension
    import os.path
    out_img = os.path.splitext(out_img)[0]

    nboxes = ctypes.c_int(0)
    alphabet = load_alphabet()

    network_predict_image(net, im)
    # Detected bounding boxes use coordinates / lengths relative to image dimensions
    relative = 1
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, relative, ctypes.pointer(nboxes))
    nboxes = nboxes.value # Cast ctypes.c_int to int

    if (nms): do_nms_obj(dets, nboxes, meta.classes, nms);
    draw_detections(im, dets, nboxes, thresh, meta.names, alphabet, meta.classes)
    if (out_img):
        save_image(im, out_img)
    else:
        save_image(im, b"predictions")

    res = []
    for j in range(nboxes):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i].decode('utf-8'), dets[j].prob[i], (b.x * im.w, b.y * im.h, b.w * im.w, b.h * im.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, nboxes)
    return res

def init_network(layers):
    """ Initialize a Darknet network using layers

    The supplied layers must be Darknet compatible. Note that actiation and
    batch_normalization layers are deprecated and should be passed as
    arguments to the previous layer. See

    :param layers: List of layers.
    :return: Darknet network.
    """
    dn_network = make_network(len(layers))
    for i, layer in enumerate(layers):
        dn_network.contents.layers[i] = layer
    return dn_network

if __name__ == "__main__":
    from pprint import pprint

    # DenseNet201
    net = load_network(b"cfg/densenet201.cfg", b"densenet201.weights", 0)
    img = load_image_color(b"data/dog.jpg", 0, 0)
    meta = get_metadata(b"cfg/imagenet1k.data")
    r = classify(net, meta, img)

    print('\nModel has %d layers\n' % (net.contents.n))
    for i in range(net.contents.n):
        print('%-3d: %-15s layer with [%4d, %4d, %4d, %4d]' % (i, get_layer_string(net.contents.layers[i].type).decode("utf-8"), net.contents.layers[i].n, net.contents.layers[i].h, net.contents.layers[i].w, net.contents.layers[i].c))

    print('\nClassifications\n')
    pprint(r[:10])
    print('\n\n')

    free_network(net)

    # Tiny YOLO v2
    net = load_network(b"cfg/tiny-yolo.cfg", b"tiny-yolo.weights", 0)
    meta = get_metadata(b"cfg/coco.data")
    r = detect(net, meta, b"data/dog.jpg", b"dog_detect_tiny.png")

    print('\n\nModel has %d layers\n' % (net.contents.n))
    for i in range(net.contents.n):
        print('%-3d: %-15s layer with [%4d, %4d, %4d, %4d]' % (i, get_layer_string(net.contents.layers[i].type).decode("utf-8"), net.contents.layers[i].n, net.contents.layers[i].h, net.contents.layers[i].w, net.contents.layers[i].c))

    print('\nDetections\n')
    pprint(r)
    print('\n\n')

    free_network(net)

    # VGG16
    net = load_network(b"cfg/vgg-16.cfg", b"vgg-16.weights", 0)
    meta = get_metadata(b"cfg/imagenet1k.data")
    img = load_image_color(b"data/dog.jpg", 0, 0)
    r = classify(net, meta, img)

    print('\n\nModel has %d layers\n' % (net.contents.n))
    for i in range(net.contents.n):
        print('%-3d: %-15s layer with [%4d, %4d, %4d, %4d]' % (i, get_layer_string(net.contents.layers[i].type).decode("utf-8"), net.contents.layers[i].n, net.contents.layers[i].h, net.contents.layers[i].w, net.contents.layers[i].c))

    print('\nClassifications\n')
    pprint(r[:10])
    print('\n\n')

    free_network(net)

    # YOLOv3
    net = load_network(b"cfg/yolov3.cfg", b"yolov3.weights", 0)
    meta = get_metadata(b"cfg/coco.data")
    r = detect(net, meta, b"data/dog.jpg", b"dog_detect_v3.png")

    print('\n\nModel has %d layers\n' % (net.contents.n))

    for i in range(net.contents.n):
        print('%-3d: %-15s layer with [%4d, %4d, %4d, %4d]' % (i, get_layer_string(net.contents.layers[i].type).decode("utf-8"), net.contents.layers[i].n, net.contents.layers[i].h, net.contents.layers[i].w, net.contents.layers[i].c))

    print('\nDetections\n')
    pprint(r)
    print('\n\n')

    free_network(net)
