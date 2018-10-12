ONNX-Darknet API
======

#### `onnx_darknet.backend.prepare`

<details>
  <summary>Prepare an ONNX model for Darknet Backend.

  </summary>
This function converts an ONNX model to an internel representation
of the computational graph called DarknetRep and returns
the converted representation.

</details>



_params_:

`model` : The ONNX model to be converted.


`device` : The device to execute this model on.


`strict` : Whether to enforce semantic equivalence between the original model
and the converted darknet model, defaults to True (yes, enforce semantic equivalence).
Changing to False is strongly discouraged.
Currently, the strict flag only affects the behavior of MaxPool and AveragePool ops.


_returns_:

A DarknetRep class object representing the ONNX model

#### `onnx_darknet.backend_rep.DarknetRep.export_graph`

<details>
  <summary>Export backend representation to darknet cfg and weights files.

  </summary>
This function obtains the model cfg and weights corresponding to the ONNX
model associated with the backend representation and serializes
to .cfg and .weights files.

</details>



_params_:

`cfg_path` : The path to the output DN .cfg file
`weights_path` : The path to the output DN .weights file


_returns_:

none.

#### `onnx_darknet.frontend.darknet_graph_to_onnx_model`
#### Warning: Not supported yet

<details>
  <summary>Converts Darknet cfg and weights to an ONNX model

  </summary>
This function converts Darknet .cfg and .weights files to an equivalent
representation of ONNX model.

</details>



_params_:

`graph_def` : Darknet cfg and weights object.


`output` : List of string or a string specifying the name
of the output graph node.


`opset` : Opset version number, list or tuple.
Default is 0 means using latest version with domain ''.
List or tuple items should be (str domain, int version number).


`producer_name` : The name of the producer.


`graph_name` : The name of the output ONNX Graph.


`ignore_unimplemented` : Convert to ONNX model and ignore all the operators
that are not currently supported by onnx-darknet.
This is an experimental feature. By enabling this feature,
the model would not be guaranteed to match the ONNX specifications.


`optimizer_passes` : List of optimization names c.f.
https://github.com/onnx/onnx/blob/master/onnx/optimizer.py for available
optimization passes.


_returns_:

The equivalent ONNX Model Proto object.

