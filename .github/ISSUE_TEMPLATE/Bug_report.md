---
name: Bug report
about: Create a report to help us improve

---

**Describe the bug**

A clear and concise description of what the bug is.

**To Reproduce**

Please give us instructions to reproduce your problem.

A __self-contained__ piece of code that can demonstrate the problem is required.

Please do not expect us to have PyTorch, Caffe2 installed.

If a model exported from PyTorch and Caffe2 is having trouble in ONNX-DN, use the next section to attach the model.

**ONNX model file**

If applicable, attach the onnx model file in question using Gist, DropBox or Google Drive.

**Python, ONNX, ONNX-DN, Darknet version and Makefile flags**

This section can be obtained by running `get_version.py` from util folder.
 - Python version:
 - ONNX version:
 - ONNX-DN version:

Please supply the Darknet details (e.g. CUDA, CUDNN) from your Makefile config
 - Darknet version:
 - Darknet build flags:

**Additional context**

Add any other context about the problem here.
