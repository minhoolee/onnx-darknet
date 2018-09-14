"""Backend for running ONNX on Darknet

To run this, you will need to have Darknet installed as well.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass

# TODO(minhoolee): Delete this, debugging only
from pprint import pprint

import numpy as np

from onnx import defs
from onnx import numpy_helper
from onnx.backend.base import Backend
from onnx.backend.base import Device
from onnx.backend.base import namedtupledict
from onnx.helper import make_opsetid

import onnx_darknet.darknet.darknet_ctypes as dn

from onnx_darknet.backend_rep import DarknetRep
from onnx_darknet.common import attr_converter
from onnx_darknet.common import attr_translator
from onnx_darknet.common import data_type
from onnx_darknet.common import exception
from onnx_darknet.common import get_device_option
from onnx_darknet.common import supports_device as common_supports_device
from onnx_darknet.common.handler_helper import get_all_backend_handlers


# TODO: Move this into ONNX main library
class OnnxNode(object):
    """
    Reimplementation of NodeProto from ONNX, but in a form
    more convenient to work with from Python.
    """

    def __init__(self, node):
        self.name = str(node.name)
        self.op_type = str(node.op_type)
        self.domain = str(node.domain)
        self.attrs = dict([(attr.name,
                            attr_translator.translate_onnx(
                                attr.name, attr_converter.convert_onnx(attr)))
                           for attr in node.attribute])
        self.inputs = list(node.input)
        self.outputs = list(node.output)
        self.node_proto = node


class DarknetBackend(Backend):
    """ Darknet Backend for ONNX
    """

    @classmethod
    def prepare(cls, model, device='CPU', strict=True, **kwargs):
        """Prepare an ONNX model for Darknet Backend.

        This function converts an ONNX model to an internal representation
        of the computational graph called DarknetRep and returns
        the converted representation.

        :param model: The ONNX model to be converted.
        :param device: The device to execute this model on.
        :param strict: Whether to enforce semantic equivalence between the original model
          and the converted Darknet model, defaults to True (yes, enforce semantic equivalence).
          Changing to False is strongly discouraged.
          Currently, the strict flag only affects the behavior of MaxPool and AveragePool ops.

        :returns: A DarknetRep class object representing the ONNX model
        """
        super(DarknetBackend, cls).prepare(model, device, **kwargs)

        return cls.onnx_model_to_darknet_rep(model, strict)

    @classmethod
    def onnx_model_to_darknet_rep(cls, model, strict):
        """ Convert ONNX model to DarknetRep.

        :param model: ONNX ModelProto object.
        :param strict: whether to enforce semantic equivalence between the original model
          and the converted Darknet model.
        :return: DarknetRep object.
        """
        return cls._onnx_graph_to_darknet_rep(model.graph, model.opset_import, strict)

    @classmethod
    # TODO: convert this method
    # Use Darknet's network (defined in darknet.h)
    def _onnx_graph_to_darknet_rep(cls, graph_def, opset, strict):
        """ Convert ONNX graph to DarknetRep.

        :param graph_def: ONNX GraphProto object.
        :param opset: ONNX OperatorSetIdProto list.
        :param strict: whether to enforce semantic equivalence between the original model
          and the converted Darknet model.
        :return: DarknetRep object.
        """
        handlers = cls._get_handlers(opset)

        # initializer: TensorProtos representing the values to initialize
        # a given tensor.
        # initialized: A list of names of the initialized tensors.
        # if graph_def.initializer:
        #     input_dict_items = cls._onnx_initializer_to_input_dict_items(
        #         graph_def.initializer)
        #     initialized = {init.name for init in graph_def.initializer}
        # else:
        #     input_dict_items = []
        #     initialized = set()
        input_dict_items = []
        initialized = set()

        # Creating placeholders for currently unknown inputs
        for value_info in graph_def.input:
            if value_info.name in initialized:
                continue

            shape = list(
                d.dim_value if (
                    d.dim_value > 0 and d.dim_param == "") else None
                for d in value_info.type.tensor_type.shape.dim)

            x = np.empty(shape=shape,
                         dtype=data_type.onnx2np(
                             value_info.type.tensor_type.elem_type))

            input_dict_items.append((value_info.name, x))

        # tensor dict: this dictionary is a map from variable names
        # to the latest produced tensors of the given name.
        # This dictionary will get updated as we build the graph to
        # record the names of newly produced tensors.
        tensor_dict = dict(input_dict_items)

        # Store names of topologically ordered layers
        layers = list()

        # TODO(minhoolee): Debugging; remove these; just to skip
        # unimplememented ops
        unimplemented = {"Dropout", "MaxPool", "Gemm"}
        # Packaged ops directly attach to their parents
        packaged = {"Relu", "Sigmoid", "Dropout", "BatchNormalization"}
        nodes = {node.name: OnnxNode(node) for node in graph_def.node}

        # DFS and remove unimplemented operation packages
        def remove_node(onnx_node):
            parents = onnx_node.inputs
            children = [child for child, node in nodes.items()
                        if onnx_node.name in node.inputs]
            for parent in reversed(parents):
                if parent.startswith(onnx_node.name.replace('_fwd', '')):
                    parents.remove(parent)
            print('Skipping {} such that {} maps to '
                  '{}'.format(onnx_node.name, parents, children))
            for parent in parents:
                nodes[parent].outputs = children
            for child in children:
                # Replace child's input without losing other child inputs
                if nodes[child].op_type in packaged or nodes[child].op_type in unimplemented:
                    remove_node(nodes[child])
                index = nodes[child].inputs.index(onnx_node.name)
                nodes[child].inputs[index: index + 1] = parents

        # Call Darknet ops corresponding to ONNX nodes and map ONNX node
        # outputs to Darknet op return values
        for node in graph_def.node:
            # onnx_node = OnnxNode(node)
            onnx_node = nodes[node.name]

            # Skip unimplemented nodes
            if onnx_node.op_type in unimplemented:
                remove_node(onnx_node)
                continue

            name = onnx_node.name + " (" + onnx_node.op_type + ")"
            print(('=' * 20 + ' {} ' + '=' * 20 + '\n').format(name))

            output_ops = cls._onnx_node_to_darknet_op(
                onnx_node, tensor_dict, handlers, opset=opset, strict=strict)
            curr_node_output_map = dict(zip(onnx_node.outputs, output_ops))
            layers.extend([name
                           for name, op in zip(onnx_node.outputs, output_ops)
                           if isinstance(op, dn.layer)])
            tensor_dict.update(curr_node_output_map)

            print(('=' * 20 + '{}' + '=' * 20 + '\n').format('=' * (len(name) + 2)))

        print()
        print()
        pprint(layers)
        # for layer in layers:
        #     pprint("Layer {}: {}".format(layer, tensor_dict[layer]))

        print("Initializing network")
        dn_net = dn.init_network([tensor_dict[layer] for layer in layers])

        dn_rep = DarknetRep()
        dn_rep.graph = dn_net
        dn_rep.name = graph_def.name
        dn_rep.inputs = [
            value_info.name
            for value_info in graph_def.input
            if value_info.name not in initialized
        ]
        dn_rep.outputs = [value_info.name for value_info in graph_def.output]
        dn_rep.tensor_dict = tensor_dict
        return dn_rep

    @classmethod
    def run_node(cls, node, inputs, device='CPU', outputs_info=None, **kwargs):
        """ Run ONNX node.

        :param node: ONNX NodeProto object.
        :param inputs: Inputs.
        :param device: Device run on.
        :param outputs_info: None.
        :param kwargs: Other args.
        :return: Outputs.
        """
        super(DarknetBackend, cls).run_node(node, inputs, device)
        node_graph = tf.Graph()
        with node_graph.as_default():
            node = OnnxNode(node)
            device_option = get_device_option(Device(device))
            input_tensors = []
            for i in inputs:
                input_tensors.append(tf.constant(i))

            if isinstance(inputs, dict):
                feed_dict_raw = inputs
            else:
                assert len(node.inputs) == len(inputs)
                feed_dict_raw = dict(zip(node.inputs, inputs))

            # TODO: is constant the best way for feeding inputs?
            input_dict = dict(
                [(x[0], tf.constant(x[1])) for x in feed_dict_raw.items()])
            ops = cls._onnx_node_to_darknet_op(node, input_dict)

            with tf.Session() as sess:
                with tf.device(device_option):
                    sess.run(tf.global_variables_initializer())
                    output_vals = sess.run(ops)

        return namedtupledict('Outputs', node.outputs)(*output_vals)

    @classmethod
    def _onnx_initializer_to_input_dict_items(cls, initializer):
        """ Convert ONNX graph initializer to input dict items.

        :param initializer: ONNX graph initializer, list of TensorProto.
        :return: List of input dict items.
        """

        def tensor2list(onnx_tensor):
            # Use the onnx.numpy_helper because the data may be raw
            # TODO(minhoolee): figure out if flatten() should be removed
            return numpy_helper.to_array(onnx_tensor).tolist()

        # return [(init.name,
        #          tf.constant(
        #              tensor2list(init),
        #              shape=init.dims,
        #              dtype=data_type.onnx2tf(init.data_type)))
        #         for init in initializer]

        return [(init.name, np.array(tensor2list(init))) for init in initializer]

    @classmethod
    def _onnx_node_to_darknet_op(cls,
                                 node,
                                 tensor_dict,
                                 handlers=None,
                                 opset=None,
                                 strict=True):
        """
        Convert onnx node to Darknet op.

        Args:
          node: Onnx node object.
          tensor_dict: Tensor dict of graph.
          opset: Opset version of the operator set. Default 0 means using latest version.
          strict: whether to enforce semantic equivalence between the original model
            and the converted Darknet model, defaults to True (yes, enforce semantic equivalence).
            Changing to False is strongly discouraged.
        Returns:
          Darknet op
        """
        handlers = handlers or cls._get_handlers(opset)
        handler = handlers[node.domain].get(node.op_type, None)
        if handler:
            return handler.handle(node, tensor_dict=tensor_dict, strict=strict)
        else:
            exception.OP_UNIMPLEMENTED_EXCEPT(node.op_type)

    @classmethod
    def _get_handlers(cls, opset):
        """ Get all backend handlers with opset.

        :param opset: ONNX OperatorSetIdProto list.
        :return: All backend handlers.
        """
        opset = opset or [make_opsetid(
            defs.ONNX_DOMAIN, defs.onnx_opset_version())]
        opset_dict = dict([(o.domain, o.version) for o in opset])
        return get_all_backend_handlers(opset_dict)

    @classmethod
    def supports_device(cls, device):
        return common_supports_device(device)


prepare = DarknetBackend.prepare

run_node = DarknetBackend.run_node

run_model = DarknetBackend.run_model

supports_device = DarknetBackend.supports_device
