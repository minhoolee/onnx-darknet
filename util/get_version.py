from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import onnx
import onnx_darknet
import pkg_resources

print("Python version:")
print(sys.version)
print("ONNX version:")
print(onnx.version.version)
print("ONNX-DN version:")
print(pkg_resources.get_distribution("onnx-dn").version)
