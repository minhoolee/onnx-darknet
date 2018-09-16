### Operations (sorted by alphabetical)

| Supported |   Darknet Name   | Darknet Alias |     ONNX Alias     |
| :-------: | :--------------: | :-----------: |     :--------:     |
|     Y     |    activation    |               |                    |
|     Y     |     avgpool      |      avg      |    AveragePool     |
|     Y     |    batchnorm     |               | BatchNormalization |
|     Y     |    connected     |     conn      |        Gemm        |
|     Y     |  convolutional   |     conv      |        Conv        |
|     N     |       cost       |               |                    |
|     N     |       crnn       |               |                    |
|     Y     |       crop       |               |       Crop*        |
|     Y     | deconvolutional  |    deconv     |   ConvTranspose    |
|    N**    |    detection     |               |                    |
|     Y     |     dropout      |               |      Dropout       |
|     Y     |       gru        |               |        GRU         |
|   N***    |       iseg       |               |                    |
|   N***    |      l2norm      |               |                    |
|     N     |      local       |               |                    |
|   N***    |     logistic     |               |                    |
|     Y     |  normalization   |      lrn      |        LRN         |
|     Y     |       lstm       |               |        LSTM        |
|     Y     |     maxpool      |      max      |      MaxPool       |
|     N     |     network      |      net      |                    |
|    N**    |      region      |               |                    |
|     N     |      reorg       |               |                    |
|     Y     |       rnn        |               |        RNN         |
|     Y     |      route       |               |       Concat       |
|     N     |     shortcut     |               |                    |
|     Y     |     softmax      |     soft      |      Softmax       |
|     Y     |     upsample     |               |      Upsample      |
|    N**    |       yolo       |               |                    |
|     Y     |                  |               |    ImageScaler*    |


### Operations (sorted by supported)

| Supported |   Darknet Name   | Darknet Alias |     ONNX Alias     |
| :-------: | :--------------: | :-----------: |     :--------:     |
|     Y     |                  |               |    ImageScaler*    |
|     Y     |       crop       |               |       Crop*        |
|     Y     |       gru        |               |        GRU         |
|     Y     |       lstm       |               |        LSTM        |
|     Y     |       rnn        |               |        RNN         |
|     Y     |      route       |               |       Concat       |
|     Y     |     avgpool      |      avg      |    AveragePool     |
|     Y     |     dropout      |               |      Dropout       |
|     Y     |     maxpool      |      max      |      MaxPool       |
|     Y     |     softmax      |     soft      |      Softmax       |
|     Y     |     upsample     |               |      Upsample      |
|     Y     |    activation    |               |                    |
|     Y     |    batchnorm     |               | BatchNormalization |
|     Y     |    connected     |     conn      |        Gemm        |
|     Y     |  convolutional   |     conv      |        Conv        |
|     Y     |  normalization   |      lrn      |        LRN         |
|     Y     | deconvolutional  |    deconv     |   ConvTranspose    |
|     N     |       cost       |               |                    |
|     N     |       crnn       |               |                    |
|     N     |      local       |               |                    |
|     N     |      reorg       |               |                    |
|     N     |     network      |      net      |                    |
|     N     |     shortcut     |               |                    |
|    N**    |       yolo       |               |                    |
|    N**    |      region      |               |                    |
|    N**    |    detection     |               |                    |
|   N***    |       iseg       |               |                    |
|   N***    |      l2norm      |               |                    |
|   N***    |     logistic     |               |                    |

\* means experimental

** means YOLO

*** means don't need to implement, very esoteric

### Activations

| Supported |   Darknet Name   |  ONNX Alias   |
| :-------: | :--------------: | :-----------: |
|     Y     |       elu        |      ELU      |
|     Y     |       relu       |     Relu      |
|     Y     |       tanh       |     Tanh      |
|     Y     |      leaky       |   LeakyRelu   |
|     Y     |     logistic     |    Sigmoid    |
|     N     |       plse       |               |
|     N     |       ramp       |               |
|     N     |      lhtan       |               |
|     N     |      linear      |               |
|     N     |      loggy       |               |
|     N     |      relie       |               |
|     N     |      stair       |               |
|     N     |     hardtan      |               |


### Method headers

```c
network *make_network(int n)

// First round (most important)
layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam)
layer make_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam)
layer make_dropout_layer(int batch, int inputs, float probability)
layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding)
layer make_softmax_layer(int batch, int inputs, int groups)

layer make_crop_layer(int batch, int h, int w, int c, int crop_height, int crop_width, int flip, float angle, float saturation, float exposure)

// Simplified method headers (remove typedef alias)
layer make_activation_layer(int batch, int inputs, ACTIVATION activation)
layer make_avgpool_layer(int batch, int w, int h, int c)
layer make_batchnorm_layer(int batch, int w, int h, int c)
layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam)
layer make_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam)
layer make_crop_layer(int batch, int h, int w, int c, int crop_height, int crop_width, int flip, float angle, float saturation, float exposure)
layer make_deconvolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int adam)
layer make_dropout_layer(int batch, int inputs, float probability)
layer make_gru_layer(int batch, int inputs, int outputs, int steps, int batch_normalize, int adam)
layer make_normalization_layer(int batch, int w, int h, int c, int size, float alpha, float beta, float kappa)
layer make_lstm_layer(int batch, int inputs, int outputs, int steps, int batch_normalize, int adam)
layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding)
layer make_rnn_layer(int batch, int inputs, int outputs, int steps, ACTIVATION activation, int batch_normalize, int adam)
layer make_route_layer(int batch, int n, int *input_layers, int *input_sizes)
layer make_softmax_layer(int batch, int inputs, int groups)
layer make_upsample_layer(int batch, int w, int h, int c, int stride)

// Actual method headers
layer make_activation_layer(int batch, int inputs, ACTIVATION activation)
avgpool_layer make_avgpool_layer(int batch, int w, int h, int c)
layer make_batchnorm_layer(int batch, int w, int h, int c)
layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, int batch_normalize, int adam)
convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam)
crop_layer make_crop_layer(int batch, int h, int w, int c, int crop_height, int crop_width, int flip, float angle, float saturation, float exposure)
layer make_deconvolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int adam)
dropout_layer make_dropout_layer(int batch, int inputs, float probability)
layer make_gru_layer(int batch, int inputs, int outputs, int steps, int batch_normalize, int adam)
layer make_normalization_layer(int batch, int w, int h, int c, int size, float alpha, float beta, float kappa)
layer make_lstm_layer(int batch, int inputs, int outputs, int steps, int batch_normalize, int adam)
maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding)
layer make_rnn_layer(int batch, int inputs, int outputs, int steps, ACTIVATION activation, int batch_normalize, int adam)
route_layer make_route_layer(int batch, int n, int *input_layers, int *input_sizes)
softmax_layer make_softmax_layer(int batch, int inputs, int groups)
layer make_upsample_layer(int batch, int w, int h, int c, int stride)
```

### Steps
#### Creating layers
- Following layers are special
  - Activation: Darknet activation is string passed to previous layer but ONNX activation is separate Node
  - Dropout: Darknet dropout is applied to previous layer of network (and takes previous layer's output and delta)
  - Batchnorm: Darknet batchnorm is also attribute of previous layer
  - Route: Looks at layer(s) offset by int value(s) and potentially combines them into one layer

#### ONNX -> DarknetRep
1. Intialize network C object
2. Parse network options ([net] and gpu_index)
3. Make params C object
4. Set params members (N, C, H, W, inputs, batch, time_steps, net)
5. Create layer C objects (and allocate space for them in net)
6. Initialize layer C objects (set weights)
7. Add layers to network

#### DarknetRep -> Darknet (export .cfg and .weights)
1. Write save_network() that somehow serializes network to string
2. Call save_network() using network C object (all [net] fields must be valid)
3. Call save_weights() using network C object

### Timeline
1. Convert tiny yolo v2 model (ONNX format) to DarknetRep
2. Convert tiny yolo v2 model (ONNX format) to DarknetRep to Darknet
3. Convert tiny yolo v2 model (Darknet format) to ONNX format and back to Darknet format
4. Figure out preprocessing for models like VGG (cropping, mean subtracting, etc.)
5. Add additional ONNX unsupported ops like region and yolo
6. Add additional ONNX supported ops like GRU, LSTM, AvgPool
7. Abstractify input and outputs to be tensors?
8. Use pybind11 or cython instead of ctypes?
9. Add more preprocessing ops (supported and unsupported)
10. Allow training
