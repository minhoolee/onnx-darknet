# Darknet Neural Network Backend and Frontend for ONNX
[![Build Status](https://travis-ci.org/minhoolee/onnx-darknet.svg?branch=master)](https://travis-ci.org/minhoolee/onnx-darknet)

### This project is still in the early stages of development so please revisit later.

[ONNX-Darknet API](https://github.com/minhoolee/onnx-darknet/blob/master/doc/API.md)

[ONNX-Darknet Op Coverage Status](https://github.com/minhoolee/onnx-darknet/blob/master/doc/support_status.md)

## Tutorials:
[Running an ONNX model using Darknet](https://github.com/onnx/tutorials/blob/master/tutorials/OnnxDarknetImport.ipynb)

[Exporting a Darknet Model to ONNX](https://github.com/onnx/tutorials/blob/master/tutorials/OnnxDarknetExport.ipynb)

## To install:
ONNX-DN requires ONNX (Open Neural Network Exchange) as an external dependency, for any issues related to ONNX installation, we refer our users to [ONNX project repository](https://github.com/onnx/onnx) for documentation and help. Notably, please ensure that protoc is available if you plan to install ONNX via pip.

The specific ONNX release version that we support in the master branch of ONNX-DN can be found [here](https://github.com/minhoolee/onnx-darknet/blob/master/ONNX_VERSION_NUMBER). This information about ONNX version requirement is automatically encoded in `setup.py`, therefore users needn't worry about ONNX version requirement when installing ONNX-DN.

To install the latest version of ONNX-DN via pip, run `pip install onnx-dn`.

Because users often have their own preferences for which variant of Darknet to install (i.e., a GPU version instead of a CPU version), we do not explicitly require Darknet in the installation script. It is therefore users' responsibility to ensure that the proper variant of Darknet is available to ONNX-DN.

## To test:
For backend, run `python -m unittest discover test`.

## Example:
In this example, we will define and run a Relu node and print the result.
This example is available as a python script at example/relu.py .
```python
from onnx_darknet.backend import run_node
from onnx import helper

node_def = helper.make_node("Relu", ["X"], ["Y"])
output = run_node(node_def, [[-0.1, 0.1]])
print(output["Y"])
```
The result is `[ 0.   0.1]`

## Development Install:
- Install ONNX master branch from source.
- Install Darknet>=1.5.0.
- Run `git clone https://github.com/minhoolee/onnx-darknet.git && cd onnx-darknet`.
- Run `pip install -e .`.

## Folder Structure:
- __onnx_darknet__ main source code file.
- __test__ test files.

## Code Standard:
- Format code:
```
pip install yapf
yapf -rip --style="{based_on_style: google, indent_width: 2}" $FilePath$
```
- Install pylint:
```
pip install pylint
wget -O /tmp/pylintrc https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/ci_build/pylintrc
```
- Check format:
```
pylint --rcfile=/tmp/pylintrc myfile.py
```

## Documentation Standard:
http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html

## Test Help:
https://docs.python.org/2/library/unittest.html

## Authors:
Mark Lee

## Thanks:
Significant contributions from [onnx-tensorflow](https://github.com/onnx/onnx-tensorflow) team made it possible to implement onnx-darknet
