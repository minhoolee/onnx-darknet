# -*- coding: utf-8 -*-
#
# TARGET arch is: []
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes


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

c_int128 = ctypes.c_ubyte*16
c_uint128 = c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 16:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*16



class struct_cudnnContext(ctypes.Structure):
    pass

cudnnHandle_t = POINTER_T(struct_cudnnContext)

# values for enumeration 'c__EA_cudnnStatus_t'
CUDNN_STATUS_SUCCESS = 0
CUDNN_STATUS_NOT_INITIALIZED = 1
CUDNN_STATUS_ALLOC_FAILED = 2
CUDNN_STATUS_BAD_PARAM = 3
CUDNN_STATUS_INTERNAL_ERROR = 4
CUDNN_STATUS_INVALID_VALUE = 5
CUDNN_STATUS_ARCH_MISMATCH = 6
CUDNN_STATUS_MAPPING_ERROR = 7
CUDNN_STATUS_EXECUTION_FAILED = 8
CUDNN_STATUS_NOT_SUPPORTED = 9
CUDNN_STATUS_LICENSE_ERROR = 10
CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING = 11
CUDNN_STATUS_RUNTIME_IN_PROGRESS = 12
CUDNN_STATUS_RUNTIME_FP_OVERFLOW = 13
c__EA_cudnnStatus_t = ctypes.c_int # enum
cudnnStatus_t = c__EA_cudnnStatus_t
class struct_cudnnRuntimeTag_t(ctypes.Structure):
    pass

cudnnRuntimeTag_t = struct_cudnnRuntimeTag_t

# values for enumeration 'c__EA_cudnnErrQueryMode_t'
CUDNN_ERRQUERY_RAWCODE = 0
CUDNN_ERRQUERY_NONBLOCKING = 1
CUDNN_ERRQUERY_BLOCKING = 2
c__EA_cudnnErrQueryMode_t = ctypes.c_int # enum
cudnnErrQueryMode_t = c__EA_cudnnErrQueryMode_t

# values for enumeration 'libraryPropertyType_t'
MAJOR_VERSION = 0
MINOR_VERSION = 1
PATCH_LEVEL = 2
libraryPropertyType_t = ctypes.c_int # enum
libraryPropertyType = libraryPropertyType_t
class struct_cudnnTensorStruct(ctypes.Structure):
    pass

cudnnTensorDescriptor_t = POINTER_T(struct_cudnnTensorStruct)
class struct_cudnnConvolutionStruct(ctypes.Structure):
    pass

cudnnConvolutionDescriptor_t = POINTER_T(struct_cudnnConvolutionStruct)
class struct_cudnnPoolingStruct(ctypes.Structure):
    pass

cudnnPoolingDescriptor_t = POINTER_T(struct_cudnnPoolingStruct)
class struct_cudnnFilterStruct(ctypes.Structure):
    pass

cudnnFilterDescriptor_t = POINTER_T(struct_cudnnFilterStruct)
class struct_cudnnLRNStruct(ctypes.Structure):
    pass

cudnnLRNDescriptor_t = POINTER_T(struct_cudnnLRNStruct)
class struct_cudnnActivationStruct(ctypes.Structure):
    pass

cudnnActivationDescriptor_t = POINTER_T(struct_cudnnActivationStruct)
class struct_cudnnSpatialTransformerStruct(ctypes.Structure):
    pass

cudnnSpatialTransformerDescriptor_t = POINTER_T(struct_cudnnSpatialTransformerStruct)
class struct_cudnnOpTensorStruct(ctypes.Structure):
    pass

cudnnOpTensorDescriptor_t = POINTER_T(struct_cudnnOpTensorStruct)
class struct_cudnnReduceTensorStruct(ctypes.Structure):
    pass

cudnnReduceTensorDescriptor_t = POINTER_T(struct_cudnnReduceTensorStruct)
class struct_cudnnCTCLossStruct(ctypes.Structure):
    pass

cudnnCTCLossDescriptor_t = POINTER_T(struct_cudnnCTCLossStruct)

# values for enumeration 'c__EA_cudnnDataType_t'
CUDNN_DATA_FLOAT = 0
CUDNN_DATA_DOUBLE = 1
CUDNN_DATA_HALF = 2
CUDNN_DATA_INT8 = 3
CUDNN_DATA_INT32 = 4
CUDNN_DATA_INT8x4 = 5
CUDNN_DATA_UINT8 = 6
CUDNN_DATA_UINT8x4 = 7
c__EA_cudnnDataType_t = ctypes.c_int # enum
cudnnDataType_t = c__EA_cudnnDataType_t

# values for enumeration 'c__EA_cudnnMathType_t'
CUDNN_DEFAULT_MATH = 0
CUDNN_TENSOR_OP_MATH = 1
c__EA_cudnnMathType_t = ctypes.c_int # enum
cudnnMathType_t = c__EA_cudnnMathType_t

# values for enumeration 'c__EA_cudnnNanPropagation_t'
CUDNN_NOT_PROPAGATE_NAN = 0
CUDNN_PROPAGATE_NAN = 1
c__EA_cudnnNanPropagation_t = ctypes.c_int # enum
cudnnNanPropagation_t = c__EA_cudnnNanPropagation_t

# values for enumeration 'c__EA_cudnnDeterminism_t'
CUDNN_NON_DETERMINISTIC = 0
CUDNN_DETERMINISTIC = 1
c__EA_cudnnDeterminism_t = ctypes.c_int # enum
cudnnDeterminism_t = c__EA_cudnnDeterminism_t

# values for enumeration 'c__EA_cudnnTensorFormat_t'
CUDNN_TENSOR_NCHW = 0
CUDNN_TENSOR_NHWC = 1
CUDNN_TENSOR_NCHW_VECT_C = 2
c__EA_cudnnTensorFormat_t = ctypes.c_int # enum
cudnnTensorFormat_t = c__EA_cudnnTensorFormat_t

# values for enumeration 'c__EA_cudnnOpTensorOp_t'
CUDNN_OP_TENSOR_ADD = 0
CUDNN_OP_TENSOR_MUL = 1
CUDNN_OP_TENSOR_MIN = 2
CUDNN_OP_TENSOR_MAX = 3
CUDNN_OP_TENSOR_SQRT = 4
CUDNN_OP_TENSOR_NOT = 5
c__EA_cudnnOpTensorOp_t = ctypes.c_int # enum
cudnnOpTensorOp_t = c__EA_cudnnOpTensorOp_t

# values for enumeration 'c__EA_cudnnReduceTensorOp_t'
CUDNN_REDUCE_TENSOR_ADD = 0
CUDNN_REDUCE_TENSOR_MUL = 1
CUDNN_REDUCE_TENSOR_MIN = 2
CUDNN_REDUCE_TENSOR_MAX = 3
CUDNN_REDUCE_TENSOR_AMAX = 4
CUDNN_REDUCE_TENSOR_AVG = 5
CUDNN_REDUCE_TENSOR_NORM1 = 6
CUDNN_REDUCE_TENSOR_NORM2 = 7
CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS = 8
c__EA_cudnnReduceTensorOp_t = ctypes.c_int # enum
cudnnReduceTensorOp_t = c__EA_cudnnReduceTensorOp_t

# values for enumeration 'c__EA_cudnnReduceTensorIndices_t'
CUDNN_REDUCE_TENSOR_NO_INDICES = 0
CUDNN_REDUCE_TENSOR_FLATTENED_INDICES = 1
c__EA_cudnnReduceTensorIndices_t = ctypes.c_int # enum
cudnnReduceTensorIndices_t = c__EA_cudnnReduceTensorIndices_t

# values for enumeration 'c__EA_cudnnIndicesType_t'
CUDNN_32BIT_INDICES = 0
CUDNN_64BIT_INDICES = 1
CUDNN_16BIT_INDICES = 2
CUDNN_8BIT_INDICES = 3
c__EA_cudnnIndicesType_t = ctypes.c_int # enum
cudnnIndicesType_t = c__EA_cudnnIndicesType_t

# values for enumeration 'c__EA_cudnnConvolutionMode_t'
CUDNN_CONVOLUTION = 0
CUDNN_CROSS_CORRELATION = 1
c__EA_cudnnConvolutionMode_t = ctypes.c_int # enum
cudnnConvolutionMode_t = c__EA_cudnnConvolutionMode_t

# values for enumeration 'c__EA_cudnnConvolutionFwdPreference_t'
CUDNN_CONVOLUTION_FWD_NO_WORKSPACE = 0
CUDNN_CONVOLUTION_FWD_PREFER_FASTEST = 1
CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = 2
c__EA_cudnnConvolutionFwdPreference_t = ctypes.c_int # enum
cudnnConvolutionFwdPreference_t = c__EA_cudnnConvolutionFwdPreference_t

# values for enumeration 'c__EA_cudnnConvolutionFwdAlgo_t'
CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0
CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1
CUDNN_CONVOLUTION_FWD_ALGO_GEMM = 2
CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = 3
CUDNN_CONVOLUTION_FWD_ALGO_FFT = 4
CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING = 5
CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD = 6
CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED = 7
CUDNN_CONVOLUTION_FWD_ALGO_COUNT = 8
c__EA_cudnnConvolutionFwdAlgo_t = ctypes.c_int # enum
cudnnConvolutionFwdAlgo_t = c__EA_cudnnConvolutionFwdAlgo_t
class struct_c_cudnnDOTh_S_cudnnDOTh_31790(ctypes.Structure):
    _pack_ = True # source:False
    _fields_ = [
    ('PADDING_0', ctypes.c_ubyte),
     ]

cudnnConvolutionFwdAlgoPerf_t = struct_c_cudnnDOTh_S_cudnnDOTh_31790

# values for enumeration 'c__EA_cudnnConvolutionBwdFilterPreference_t'
CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE = 0
CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST = 1
CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT = 2
c__EA_cudnnConvolutionBwdFilterPreference_t = ctypes.c_int # enum
cudnnConvolutionBwdFilterPreference_t = c__EA_cudnnConvolutionBwdFilterPreference_t

# values for enumeration 'c__EA_cudnnConvolutionBwdFilterAlgo_t'
CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = 0
CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 = 1
CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT = 2
CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 = 3
CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD = 4
CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED = 5
CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING = 6
CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT = 7
c__EA_cudnnConvolutionBwdFilterAlgo_t = ctypes.c_int # enum
cudnnConvolutionBwdFilterAlgo_t = c__EA_cudnnConvolutionBwdFilterAlgo_t
class struct_c_cudnnDOTh_S_cudnnDOTh_40612(ctypes.Structure):
    _pack_ = True # source:False
    _fields_ = [
    ('PADDING_0', ctypes.c_ubyte),
     ]

cudnnConvolutionBwdFilterAlgoPerf_t = struct_c_cudnnDOTh_S_cudnnDOTh_40612

# values for enumeration 'c__EA_cudnnConvolutionBwdDataPreference_t'
CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE = 0
CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST = 1
CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT = 2
c__EA_cudnnConvolutionBwdDataPreference_t = ctypes.c_int # enum
cudnnConvolutionBwdDataPreference_t = c__EA_cudnnConvolutionBwdDataPreference_t

# values for enumeration 'c__EA_cudnnConvolutionBwdDataAlgo_t'
CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = 0
CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 = 1
CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT = 2
CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING = 3
CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD = 4
CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED = 5
CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT = 6
c__EA_cudnnConvolutionBwdDataAlgo_t = ctypes.c_int # enum
cudnnConvolutionBwdDataAlgo_t = c__EA_cudnnConvolutionBwdDataAlgo_t
class struct_c_cudnnDOTh_S_cudnnDOTh_47320(ctypes.Structure):
    _pack_ = True # source:False
    _fields_ = [
    ('PADDING_0', ctypes.c_ubyte),
     ]

cudnnConvolutionBwdDataAlgoPerf_t = struct_c_cudnnDOTh_S_cudnnDOTh_47320

# values for enumeration 'c__EA_cudnnSoftmaxAlgorithm_t'
CUDNN_SOFTMAX_FAST = 0
CUDNN_SOFTMAX_ACCURATE = 1
CUDNN_SOFTMAX_LOG = 2
c__EA_cudnnSoftmaxAlgorithm_t = ctypes.c_int # enum
cudnnSoftmaxAlgorithm_t = c__EA_cudnnSoftmaxAlgorithm_t

# values for enumeration 'c__EA_cudnnSoftmaxMode_t'
CUDNN_SOFTMAX_MODE_INSTANCE = 0
CUDNN_SOFTMAX_MODE_CHANNEL = 1
c__EA_cudnnSoftmaxMode_t = ctypes.c_int # enum
cudnnSoftmaxMode_t = c__EA_cudnnSoftmaxMode_t

# values for enumeration 'c__EA_cudnnPoolingMode_t'
CUDNN_POOLING_MAX = 0
CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1
CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2
CUDNN_POOLING_MAX_DETERMINISTIC = 3
c__EA_cudnnPoolingMode_t = ctypes.c_int # enum
cudnnPoolingMode_t = c__EA_cudnnPoolingMode_t

# values for enumeration 'c__EA_cudnnActivationMode_t'
CUDNN_ACTIVATION_SIGMOID = 0
CUDNN_ACTIVATION_RELU = 1
CUDNN_ACTIVATION_TANH = 2
CUDNN_ACTIVATION_CLIPPED_RELU = 3
CUDNN_ACTIVATION_ELU = 4
CUDNN_ACTIVATION_IDENTITY = 5
c__EA_cudnnActivationMode_t = ctypes.c_int # enum
cudnnActivationMode_t = c__EA_cudnnActivationMode_t

# values for enumeration 'c__EA_cudnnLRNMode_t'
CUDNN_LRN_CROSS_CHANNEL_DIM1 = 0
c__EA_cudnnLRNMode_t = ctypes.c_int # enum
cudnnLRNMode_t = c__EA_cudnnLRNMode_t

# values for enumeration 'c__EA_cudnnDivNormMode_t'
CUDNN_DIVNORM_PRECOMPUTED_MEANS = 0
c__EA_cudnnDivNormMode_t = ctypes.c_int # enum
cudnnDivNormMode_t = c__EA_cudnnDivNormMode_t

# values for enumeration 'c__EA_cudnnBatchNormMode_t'
CUDNN_BATCHNORM_PER_ACTIVATION = 0
CUDNN_BATCHNORM_SPATIAL = 1
CUDNN_BATCHNORM_SPATIAL_PERSISTENT = 2
c__EA_cudnnBatchNormMode_t = ctypes.c_int # enum
cudnnBatchNormMode_t = c__EA_cudnnBatchNormMode_t

# values for enumeration 'c__EA_cudnnSamplerType_t'
CUDNN_SAMPLER_BILINEAR = 0
c__EA_cudnnSamplerType_t = ctypes.c_int # enum
cudnnSamplerType_t = c__EA_cudnnSamplerType_t
class struct_cudnnDropoutStruct(ctypes.Structure):
    pass

cudnnDropoutDescriptor_t = POINTER_T(struct_cudnnDropoutStruct)

# values for enumeration 'c__EA_cudnnRNNMode_t'
CUDNN_RNN_RELU = 0
CUDNN_RNN_TANH = 1
CUDNN_LSTM = 2
CUDNN_GRU = 3
c__EA_cudnnRNNMode_t = ctypes.c_int # enum
cudnnRNNMode_t = c__EA_cudnnRNNMode_t

# values for enumeration 'c__EA_cudnnDirectionMode_t'
CUDNN_UNIDIRECTIONAL = 0
CUDNN_BIDIRECTIONAL = 1
c__EA_cudnnDirectionMode_t = ctypes.c_int # enum
cudnnDirectionMode_t = c__EA_cudnnDirectionMode_t

# values for enumeration 'c__EA_cudnnRNNInputMode_t'
CUDNN_LINEAR_INPUT = 0
CUDNN_SKIP_INPUT = 1
c__EA_cudnnRNNInputMode_t = ctypes.c_int # enum
cudnnRNNInputMode_t = c__EA_cudnnRNNInputMode_t

# values for enumeration 'c__EA_cudnnRNNAlgo_t'
CUDNN_RNN_ALGO_STANDARD = 0
CUDNN_RNN_ALGO_PERSIST_STATIC = 1
CUDNN_RNN_ALGO_PERSIST_DYNAMIC = 2
CUDNN_RNN_ALGO_COUNT = 3
c__EA_cudnnRNNAlgo_t = ctypes.c_int # enum
cudnnRNNAlgo_t = c__EA_cudnnRNNAlgo_t
class struct_cudnnAlgorithmStruct(ctypes.Structure):
    pass

cudnnAlgorithmDescriptor_t = POINTER_T(struct_cudnnAlgorithmStruct)
class struct_cudnnAlgorithmPerformanceStruct(ctypes.Structure):
    pass

cudnnAlgorithmPerformance_t = POINTER_T(struct_cudnnAlgorithmPerformanceStruct)
class struct_cudnnRNNStruct(ctypes.Structure):
    pass

cudnnRNNDescriptor_t = POINTER_T(struct_cudnnRNNStruct)
class struct_cudnnPersistentRNNPlan(ctypes.Structure):
    pass

cudnnPersistentRNNPlan_t = POINTER_T(struct_cudnnPersistentRNNPlan)

# values for enumeration 'c__EA_cudnnCTCLossAlgo_t'
CUDNN_CTC_LOSS_ALGO_DETERMINISTIC = 0
CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC = 1
c__EA_cudnnCTCLossAlgo_t = ctypes.c_int # enum
cudnnCTCLossAlgo_t = c__EA_cudnnCTCLossAlgo_t
class struct_c__SA_cudnnAlgorithm_t(ctypes.Structure):
    pass

class union_Algorithm(ctypes.Union):
    _pack_ = True # source:False
    _fields_ = [
    ('convFwdAlgo', cudnnConvolutionFwdAlgo_t),
    ('convBwdFilterAlgo', cudnnConvolutionBwdFilterAlgo_t),
    ('convBwdDataAlgo', cudnnConvolutionBwdDataAlgo_t),
    ('RNNAlgo', cudnnRNNAlgo_t),
    ('CTCLossAlgo', cudnnCTCLossAlgo_t),
     ]

struct_c__SA_cudnnAlgorithm_t._pack_ = True # source:False
struct_c__SA_cudnnAlgorithm_t._fields_ = [
    ('algo', union_Algorithm),
]

cudnnAlgorithm_t = struct_c__SA_cudnnAlgorithm_t

# values for enumeration 'c__EA_cudnnSeverity_t'
CUDNN_SEV_FATAL = 0
CUDNN_SEV_ERROR = 1
CUDNN_SEV_WARNING = 2
CUDNN_SEV_INFO = 3
c__EA_cudnnSeverity_t = ctypes.c_int # enum
cudnnSeverity_t = c__EA_cudnnSeverity_t
class struct_c_cudnnDOTh_S_cudnnDOTh_119709(ctypes.Structure):
    _pack_ = True # source:False
    _fields_ = [
    ('PADDING_0', ctypes.c_ubyte),
     ]

cudnnDebug_t = struct_c_cudnnDOTh_S_cudnnDOTh_119709
cudnnCallback_t = POINTER_T(ctypes.CFUNCTYPE(None, c__EA_cudnnSeverity_t, POINTER_T(None), POINTER_T(struct_c_cudnnDOTh_S_cudnnDOTh_119709), POINTER_T(ctypes.c_char)))
__all__ = \
    ['cudnnLRNDescriptor_t', 'CUDNN_SEV_FATAL', 'CUDNN_64BIT_INDICES',
    'CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING',
    'c__EA_cudnnCTCLossAlgo_t', 'CUDNN_ERRQUERY_NONBLOCKING',
    'CUDNN_DATA_FLOAT', 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT',
    'CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT', 'cudnnActivationMode_t',
    'CUDNN_RNN_ALGO_PERSIST_DYNAMIC',
    'c__EA_cudnnConvolutionBwdFilterPreference_t', 'CUDNN_DATA_INT8',
    'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0',
    'cudnnConvolutionBwdFilterPreference_t',
    'cudnnFilterDescriptor_t', 'CUDNN_ACTIVATION_TANH',
    'CUDNN_RNN_ALGO_PERSIST_STATIC', 'cudnnNanPropagation_t',
    'cudnnConvolutionBwdDataAlgo_t',
    'CUDNN_CONVOLUTION_FWD_PREFER_FASTEST', 'CUDNN_DATA_UINT8x4',
    'CUDNN_CONVOLUTION', 'c__EA_cudnnConvolutionBwdDataAlgo_t',
    'struct_c__SA_cudnnAlgorithm_t', 'CUDNN_REDUCE_TENSOR_MAX',
    'CUDNN_STATUS_ALLOC_FAILED', 'struct_cudnnPoolingStruct',
    'cudnnActivationDescriptor_t', 'cudnnDivNormMode_t',
    'c__EA_cudnnActivationMode_t', 'struct_cudnnLRNStruct',
    'PATCH_LEVEL', 'cudnnConvolutionFwdAlgo_t',
    'CUDNN_CONVOLUTION_FWD_ALGO_COUNT', 'CUDNN_SOFTMAX_LOG',
    'CUDNN_STATUS_NOT_INITIALIZED',
    'CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD',
    'cudnnConvolutionFwdAlgoPerf_t', 'c__EA_cudnnTensorFormat_t',
    'cudnnDirectionMode_t', 'c__EA_cudnnConvolutionFwdPreference_t',
    'libraryPropertyType', 'c__EA_cudnnMathType_t',
    'CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST',
    'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED',
    'CUDNN_SOFTMAX_MODE_INSTANCE', 'CUDNN_SOFTMAX_MODE_CHANNEL',
    'cudnnSoftmaxMode_t', 'CUDNN_REDUCE_TENSOR_AMAX',
    'CUDNN_TENSOR_NCHW_VECT_C', 'cudnnIndicesType_t',
    'CUDNN_BATCHNORM_SPATIAL', 'cudnnBatchNormMode_t',
    'cudnnConvolutionMode_t', 'struct_cudnnRNNStruct',
    'CUDNN_PROPAGATE_NAN', 'c__EA_cudnnLRNMode_t',
    'CUDNN_RNN_ALGO_STANDARD', 'CUDNN_STATUS_MAPPING_ERROR',
    'struct_cudnnOpTensorStruct',
    'CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED',
    'struct_c_cudnnDOTh_S_cudnnDOTh_40612',
    'struct_cudnnActivationStruct', 'CUDNN_STATUS_ARCH_MISMATCH',
    'struct_cudnnFilterStruct', 'CUDNN_8BIT_INDICES',
    'struct_cudnnAlgorithmPerformanceStruct', 'CUDNN_SEV_WARNING',
    'CUDNN_SOFTMAX_ACCURATE', 'CUDNN_STATUS_BAD_PARAM',
    'CUDNN_REDUCE_TENSOR_NO_INDICES', 'CUDNN_SEV_INFO',
    'cudnnReduceTensorOp_t', 'CUDNN_GRU', 'c__EA_cudnnErrQueryMode_t',
    'CUDNN_STATUS_RUNTIME_IN_PROGRESS',
    'struct_c_cudnnDOTh_S_cudnnDOTh_31790',
    'CUDNN_STATUS_RUNTIME_FP_OVERFLOW',
    'c__EA_cudnnConvolutionFwdAlgo_t', 'CUDNN_POOLING_MAX',
    'CUDNN_ACTIVATION_ELU', 'cudnnConvolutionBwdDataPreference_t',
    'struct_cudnnContext', 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1',
    'cudnnDebug_t', 'cudnnCTCLossDescriptor_t',
    'c__EA_cudnnOpTensorOp_t', 'CUDNN_BATCHNORM_PER_ACTIVATION',
    'cudnnStatus_t', 'CUDNN_RNN_RELU', 'cudnnErrQueryMode_t',
    'cudnnSpatialTransformerDescriptor_t', 'CUDNN_REDUCE_TENSOR_ADD',
    'cudnnOpTensorOp_t', 'CUDNN_DATA_DOUBLE',
    'cudnnPersistentRNNPlan_t', 'CUDNN_ERRQUERY_BLOCKING',
    'CUDNN_CONVOLUTION_FWD_ALGO_FFT',
    'CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING',
    'c__EA_cudnnConvolutionMode_t',
    'CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD', 'CUDNN_DEFAULT_MATH',
    'CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED',
    'CUDNN_CONVOLUTION_BWD_DATA_ALGO_0',
    'c__EA_cudnnNanPropagation_t', 'cudnnLRNMode_t',
    'cudnnAlgorithmDescriptor_t',
    'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD',
    'CUDNN_NOT_PROPAGATE_NAN', 'cudnnTensorFormat_t',
    'cudnnCTCLossAlgo_t', 'struct_cudnnRuntimeTag_t',
    'c__EA_cudnnDivNormMode_t', 'cudnnSamplerType_t',
    'CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT',
    'c__EA_cudnnSamplerType_t', 'cudnnMathType_t', 'cudnnRNNMode_t',
    'CUDNN_OP_TENSOR_MAX', 'CUDNN_DIVNORM_PRECOMPUTED_MEANS',
    'CUDNN_STATUS_LICENSE_ERROR', 'cudnnRuntimeTag_t',
    'CUDNN_REDUCE_TENSOR_MIN', 'CUDNN_LSTM',
    'c__EA_cudnnRNNInputMode_t', 'union_Algorithm',
    'CUDNN_OP_TENSOR_MIN', 'CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC',
    'CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT', 'c__EA_cudnnSeverity_t',
    'c__EA_cudnnPoolingMode_t', 'cudnnCallback_t',
    'CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE',
    'c__EA_cudnnRNNMode_t', 'c__EA_cudnnSoftmaxAlgorithm_t',
    'c__EA_cudnnStatus_t', 'CUDNN_DATA_INT32', 'cudnnHandle_t',
    'MAJOR_VERSION', 'cudnnTensorDescriptor_t',
    'c__EA_cudnnIndicesType_t', 'CUDNN_SAMPLER_BILINEAR',
    'CUDNN_SEV_ERROR', 'cudnnSoftmaxAlgorithm_t',
    'CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS',
    'CUDNN_BATCHNORM_SPATIAL_PERSISTENT', 'cudnnRNNInputMode_t',
    'CUDNN_UNIDIRECTIONAL',
    'CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT',
    'cudnnConvolutionBwdDataAlgoPerf_t', 'c__EA_cudnnSoftmaxMode_t',
    'cudnnReduceTensorIndices_t', 'cudnnPoolingDescriptor_t',
    'cudnnOpTensorDescriptor_t', 'CUDNN_TENSOR_NHWC',
    'CUDNN_CONVOLUTION_FWD_ALGO_DIRECT', 'CUDNN_ACTIVATION_IDENTITY',
    'CUDNN_RNN_ALGO_COUNT', 'MINOR_VERSION',
    'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM',
    'CUDNN_LINEAR_INPUT', 'CUDNN_ERRQUERY_RAWCODE',
    'cudnnDropoutDescriptor_t', 'cudnnConvolutionDescriptor_t',
    'CUDNN_32BIT_INDICES', 'CUDNN_CONVOLUTION_BWD_DATA_ALGO_1',
    'cudnnAlgorithm_t', 'cudnnRNNDescriptor_t',
    'CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST', 'cudnnRNNAlgo_t',
    'CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING',
    'CUDNN_STATUS_EXECUTION_FAILED',
    'CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT',
    'c__EA_cudnnReduceTensorOp_t', 'CUDNN_REDUCE_TENSOR_NORM1',
    'cudnnDataType_t', 'CUDNN_CROSS_CORRELATION',
    'CUDNN_CTC_LOSS_ALGO_DETERMINISTIC',
    'CUDNN_CONVOLUTION_FWD_ALGO_GEMM', 'CUDNN_ACTIVATION_RELU',
    'CUDNN_LRN_CROSS_CHANNEL_DIM1', 'CUDNN_REDUCE_TENSOR_NORM2',
    'CUDNN_OP_TENSOR_NOT', 'CUDNN_ACTIVATION_CLIPPED_RELU',
    'CUDNN_OP_TENSOR_MUL', 'CUDNN_OP_TENSOR_SQRT',
    'cudnnReduceTensorDescriptor_t', 'CUDNN_REDUCE_TENSOR_MUL',
    'c__EA_cudnnConvolutionBwdFilterAlgo_t',
    'struct_c_cudnnDOTh_S_cudnnDOTh_47320', 'CUDNN_DATA_HALF',
    'struct_cudnnCTCLossStruct', 'struct_cudnnTensorStruct',
    'CUDNN_STATUS_SUCCESS', 'c__EA_cudnnReduceTensorIndices_t',
    'struct_c_cudnnDOTh_S_cudnnDOTh_119709',
    'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM',
    'CUDNN_REDUCE_TENSOR_AVG',
    'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING',
    'libraryPropertyType_t', 'struct_cudnnSpatialTransformerStruct',
    'CUDNN_16BIT_INDICES', 'CUDNN_RNN_TANH',
    'c__EA_cudnnBatchNormMode_t',
    'CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE', 'CUDNN_TENSOR_NCHW',
    'struct_cudnnAlgorithmStruct',
    'c__EA_cudnnConvolutionBwdDataPreference_t',
    'struct_cudnnReduceTensorStruct', 'CUDNN_SOFTMAX_FAST',
    'CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING',
    'CUDNN_POOLING_MAX_DETERMINISTIC', 'CUDNN_TENSOR_OP_MATH',
    'CUDNN_STATUS_INTERNAL_ERROR', 'c__EA_cudnnDeterminism_t',
    'CUDNN_OP_TENSOR_ADD', 'c__EA_cudnnDataType_t',
    'c__EA_cudnnDirectionMode_t', 'CUDNN_SKIP_INPUT',
    'CUDNN_DATA_INT8x4', 'CUDNN_CONVOLUTION_FWD_NO_WORKSPACE',
    'cudnnConvolutionBwdFilterAlgo_t',
    'cudnnConvolutionBwdFilterAlgoPerf_t', 'c__EA_cudnnRNNAlgo_t',
    'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT',
    'struct_cudnnConvolutionStruct', 'cudnnDeterminism_t',
    'CUDNN_STATUS_INVALID_VALUE', 'cudnnConvolutionFwdPreference_t',
    'CUDNN_REDUCE_TENSOR_FLATTENED_INDICES', 'CUDNN_BIDIRECTIONAL',
    'CUDNN_DATA_UINT8', 'cudnnAlgorithmPerformance_t',
    'struct_cudnnPersistentRNNPlan', 'cudnnSeverity_t',
    'CUDNN_ACTIVATION_SIGMOID', 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3',
    'CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING',
    'struct_cudnnDropoutStruct', 'CUDNN_NON_DETERMINISTIC',
    'CUDNN_DETERMINISTIC', 'cudnnPoolingMode_t',
    'CUDNN_STATUS_NOT_SUPPORTED']
