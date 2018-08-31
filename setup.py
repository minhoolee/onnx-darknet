import os
import setuptools

TOP_DIR = os.path.realpath(os.path.dirname(__file__))

with open(os.path.join(TOP_DIR, 'VERSION_NUMBER')) as version_file:
  version = version_file.read().strip()

if os.getenv('TRAVIS'):
  # On travis, we install from source, therefore no need to specify version.
  onnx_dep = "onnx"
else:
  # For user, we install the onnx release known to work with our release.
  with open(os.path.join(TOP_DIR, 'ONNX_VERSION_NUMBER')) as onnx_version_file:
    onnx_version = onnx_version_file.read().strip()
    onnx_dep = "onnx>=" + onnx_version

setuptools.setup(
    name='onnx-darknet',
    version=version,
    description=
    'Darknet backend and frontend for ONNX (Open Neural Network Exchange).',
    install_requires=[onnx_dep],
    url='https://github.com/minhoolee/onnx-darknet/',
    author='Mark Lee',
    author_email='mhlee@ucsd.edu',
    license='Apache License 2.0',
    packages=setuptools.find_packages(),
    zip_safe=False)
