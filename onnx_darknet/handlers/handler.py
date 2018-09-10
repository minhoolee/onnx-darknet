from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect

from onnx import defs

from onnx_darknet.common import exception
from onnx_darknet.common import IS_PYTHON3


class Handler(object):
    """ This class is base handler class.
    Base backend and frontend base handler class inherit this class.

    All operator handler MUST put decorator @onnx_op and @darknet_op to register corresponding op.
    """

    ONNX_OP = None
    DARKNET_OP = []

    DOMAIN = defs.ONNX_DOMAIN
    VERSION = 0
    SINCE_VERSION = 0

    @classmethod
    def check_cls(cls):
        if not cls.ONNX_OP:
            raise ValueError(
                "{} doesn't have ONNX_OP. "
                "Please use Handler.onnx_op decorator to register ONNX_OP.".format(
                    cls.__name__))

    @classmethod
    def args_check(cls, node, **kwargs):
        """ Check args. e.g. if shape info is in graph.
        Raise exception if failed.

        :param node: NodeProto for backend or TensorflowNode for frontend.
        :param kwargs: Other args.
        """
        pass

    @classmethod
    def handle(cls, node, **kwargs):
        """ Main method in handler. It will find corresponding versioned handle method,
        whose name format is `version_%d`. So prefix `version_` is reserved in onnx-tensorflow.
        DON'T use it for other purpose.

        :param node: NodeProto for backend or TensorflowNode for frontend.
        :param kwargs: Other args.
        :return: NodeProto for frontend or TensorflowNode for backend.
        """
        ver_handle = getattr(cls, "version_{}".format(cls.SINCE_VERSION), None)
        if ver_handle:
            cls.args_check(node, **kwargs)
            return ver_handle(node, **kwargs)
        exception.OP_UNIMPLEMENTED_EXCEPT(node.op_type, cls.SINCE_VERSION)
        return None

    @classmethod
    def get_versions(cls):
        """ Get all support versions.

        :return: Version list.
        """
        versions = []
        for k, v in inspect.getmembers(cls, inspect.ismethod):
            if k.startswith("version_"):
                versions.append(int(k.replace("version_", "")))
        return versions

    @staticmethod
    def onnx_op(op):
        return Handler.property_register("ONNX_OP", op)

    @staticmethod
    def darknet_op(op):
        ops = op
        if not isinstance(ops, list):
            ops = [ops]
        return Handler.property_register("DARKNET_OP", ops)

    @staticmethod
    def darknet_func(func):
        return Handler.property_register("DARKNET_FUNC", func)

    @staticmethod
    def domain(d):
        return Handler.property_register("DOMAIN", d)

    @staticmethod
    def property_register(name, value):

        def deco(cls):
            if inspect.isfunction(value) and not IS_PYTHON3:
                setattr(cls, name, staticmethod(value))
            else:
                setattr(cls, name, value)
            return cls

        return deco


domain = Handler.domain

onnx_op = Handler.onnx_op

darknet_op = Handler.darknet_op

darknet_func = Handler.darknet_func

property_register = Handler.property_register
