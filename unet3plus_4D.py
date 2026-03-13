# Import standard python modules
import tensorflow as tf
import numpy as np

from tensorflow.keras import layers, models
from tensorflow.keras.layers import (Conv1D, Conv3D, MaxPooling1D, MaxPooling3D, UpSampling1D, UpSampling3D, TimeDistributed, Lambda)


def central_crop(tensor, shape):
  """Crop the central region of a tensor.

  Args:
    tensor: A `Tensor`.
    shape: A `Tensor`. The shape of the region to crop. The length of `shape`
      must be equal to or less than the rank of `tensor`. If the length of
      `shape` is less than the rank of tensor, the operation is applied along
      the last `len(shape)` dimensions of `tensor`. Any component of `shape` can
      be set to the special value -1 to leave the corresponding dimension
      unchanged.

  Returns:
    A `Tensor`. Has the same type as `tensor`. The centrally cropped tensor.

  Raises:
    ValueError: If `shape` has a rank other than 1.
  """
  tensor = tf.convert_to_tensor(tensor)
  input_shape_tensor = tf.shape(tensor)
  target_shape_tensor = tf.convert_to_tensor(shape)

  # Static checks.
  if target_shape_tensor.shape.rank != 1:
    raise ValueError(f"`shape` must have rank 1. Received: {shape}")

  # Support a target shape with less dimensions than input. In that case, the
  # target shape applies to the last dimensions of input.
  if not isinstance(shape, tf.Tensor):
    shape = [-1] * (tensor.shape.rank - len(shape)) + list(shape)
  target_shape_tensor = tf.concat([
      tf.tile([-1], [tf.rank(tensor) - tf.size(target_shape_tensor)]),
      target_shape_tensor], 0)

  # Dynamic checks.
  checks = [
      tf.debugging.assert_greater_equal(tf.rank(tensor), tf.size(shape)),
      tf.debugging.assert_less_equal(
          target_shape_tensor, tf.shape(tensor), message=(
              "Target shape cannot be greater than input shape."))
  ]
  with tf.control_dependencies(checks):
    tensor = tf.identity(tensor)

  # Crop the tensor.
  slice_begin = tf.where(
      target_shape_tensor >= 0,
      tf.math.maximum(input_shape_tensor - target_shape_tensor, 0) // 2,
      0)
  slice_size = tf.where(
      target_shape_tensor >= 0,
      tf.math.minimum(input_shape_tensor, target_shape_tensor),
      -1)
  tensor = tf.slice(tensor, slice_begin, slice_size)

  # Set static shape, if possible.
  static_shape = _compute_static_output_shape(tensor.shape, shape)
  if static_shape is not None:
    tensor = tf.ensure_shape(tensor, static_shape)

  return tensor

def _compute_static_output_shape(input_shape, target_shape):
  """Compute the static output shape of a resize operation.

  Args:
    input_shape: The static shape of the input tensor.
    target_shape: The target shape.

  Returns:
    The static output shape.
  """
  output_shape = None

  if isinstance(target_shape, tf.Tensor):
    # If target shape is a tensor, we can't infer the output shape.
    return None

  # Get static tensor shape, after replacing -1 values by `None`.
  output_shape = tf.TensorShape(
      [s if s >= 0 else None for s in target_shape])

  # Complete any unspecified target dimensions with those of the
  # input tensor, if known.
  output_shape = tf.TensorShape(
      [s_target or s_input for (s_target, s_input) in zip(
          output_shape.as_list(), input_shape.as_list())])

  return output_shape


def resize_with_crop_or_pad(tensor, shape, padding_mode='constant'):
  """Crops and/or pads a tensor to a target shape.

  Pads symmetrically or crops centrally the input tensor as necessary to achieve
  the requested shape.

  Args:
    tensor: A `Tensor`.
    shape: A `Tensor`. The shape of the output tensor. The length of `shape`
      must be equal to or less than the rank of `tensor`. If the length of
      `shape` is less than the rank of tensor, the operation is applied along
      the last `len(shape)` dimensions of `tensor`. Any component of `shape` can
      be set to the special value -1 to leave the corresponding dimension
      unchanged.
    padding_mode: A `str`. Must be one of `'constant'`, `'reflect'` or
      `'symmetric'`.

  Returns:
    A `Tensor`. Has the same type as `tensor`. The symmetrically padded/cropped
    tensor.
  """
  tensor = tf.convert_to_tensor(tensor)
  input_shape = tensor.shape
  input_shape_tensor = tf.shape(tensor)
  target_shape = shape
  target_shape_tensor = tf.convert_to_tensor(shape)

  # Support a target shape with less dimensions than input. In that case, the
  # target shape applies to the last dimensions of input.
  if not isinstance(target_shape, tf.Tensor):
    target_shape = [-1] * (input_shape.rank - len(shape)) + list(shape)
  target_shape_tensor = tf.concat([
      tf.tile([-1], [tf.rank(tensor) - tf.size(shape)]),
      target_shape_tensor], 0)

  # Dynamic checks.
  checks = [
      tf.debugging.assert_greater_equal(tf.rank(tensor),
                                        tf.size(target_shape_tensor)),
  ]
  with tf.control_dependencies(checks):
    tensor = tf.identity(tensor)

  # Pad the tensor.
  pad_left = tf.where(
      target_shape_tensor >= 0,
      tf.math.maximum(target_shape_tensor - input_shape_tensor, 0) // 2,
      0)
  pad_right = tf.where(
      target_shape_tensor >= 0,
      (tf.math.maximum(target_shape_tensor - input_shape_tensor, 0) + 1) // 2,
      0)

  tensor = tf.pad(tensor, tf.transpose(tf.stack([pad_left, pad_right])), # pylint: disable=no-value-for-parameter,unexpected-keyword-arg
                  mode=padding_mode)

  # Crop the tensor.
  tensor = central_crop(tensor, target_shape)

  static_shape = _compute_static_output_shape(input_shape, target_shape)
  if static_shape is not None:
    tensor = tf.ensure_shape(tensor, static_shape)

  return tensor

def Transpose(perm):
    return Lambda(lambda x: tf.transpose(x, perm))


def Multi_TimeDistributed(layer, iter):
    for _ in range(iter):
        layer = TimeDistributed(layer)
    return layer



class Conv4D(layers.Layer):
    def __init__(self, 
                 filters,
                 kernel_size,
                 padding="same",
                 kernel_initializer=tf.keras.initializers.HeNormal(seed=0),
                 bias_initializer=tf.keras.initializers.Zeros(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activation=None,
                 **kwargs):  # Accept other keyword args like 'name'
        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activation = activation

        self.conv3d_1 = layers.TimeDistributed(
            layers.Conv3D(filters, 
                          kernel_size, 
                          padding=padding,
                          kernel_initializer=kernel_initializer,
                          bias_initializer=bias_initializer,
                          kernel_regularizer=kernel_regularizer,
                          bias_regularizer=bias_regularizer,
                          activation=activation
                          )
        )
        
        self.conv1d_1 = Multi_TimeDistributed(
            layers.Conv1D(filters, 
                          kernel_size[0], 
                          padding=padding,
                          kernel_initializer=kernel_initializer,
                          bias_initializer=bias_initializer,
                          kernel_regularizer=kernel_regularizer,
                          bias_regularizer=bias_regularizer,
                          activation=activation
                          ),
            iter=3
        )

    def call(self, inputs):
        x = self.conv3d_1(inputs)
        # Use tf.transpose instead of Transpose layer
        x = tf.transpose(x, perm=(0, 2, 3, 4, 1, 5))
        x = self.conv1d_1(x)
        x = tf.transpose(x, perm=(0, 4, 1, 2, 3, 5))
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "padding": self.padding,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "bias_regularizer": self.bias_regularizer,
            "activation": self.activation
        })
        return config




class MaxPool4D(layers.Layer):
    def __init__(self, 
                 pool_size,
                 **kwargs):  # Accept other keyword args like 'name'
                 
        super().__init__()
        self.pool_size = pool_size
        self.pool3d = TimeDistributed(MaxPooling3D(pool_size = pool_size[1:]))
        self.time_pool = (pool_size[0],)

        self.pool1d = Multi_TimeDistributed(MaxPooling1D(pool_size = self.time_pool), iter=3)

    def call(self, inputs):
        p = self.pool3d(inputs)
        if self.time_pool[0] > 1:
          p = Transpose((0, 2, 3, 4, 1, 5))(p)
          p = self.pool1d(p)
          p = Transpose((0, 4, 1, 2, 3, 5))(p)
        return p


    def get_config(self):
        config = super().get_config()
        config.update({
            "pool_size": self.pool_size
        })
        return config
    
class UpSampling4D(layers.Layer):
    def __init__(self, 
                 size,
                 **kwargs):  # Accept other keyword args like 'name'
                 
        super().__init__()
        self.size = size

        self.upsample3d = TimeDistributed(UpSampling3D(size = size[1:]))
        self.time_size = size[0]
        self.upsample1d = Multi_TimeDistributed(UpSampling1D(size = self.time_size), iter=3)

    def call(self, inputs):
        x = self.upsample3d(inputs)
        if self.time_size > 1:
          x = Transpose((0, 2, 3, 4, 1, 5))(x)
          x = self.upsample1d(x)
          x = Transpose((0, 4, 1, 2, 3, 5))(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "size": self.size
        })
        return config



def get_nd_layer(name, rank):
  """Get an N-D layer object.

  Args:
    name: A `str`. The name of the requested layer.
    rank: An `int`. The rank of the requested layer.

  Returns:
    A `tf.keras.layers.Layer` object.

  Raises:
    ValueError: If the requested layer is unknown to TFMRI.
  """
  try:
    return _ND_LAYERS[(name, rank)]
  except KeyError as err:
    raise ValueError(
        f"Could not find a layer with name '{name}' and rank {rank}.") from err


_ND_LAYERS = {
    ('AveragePooling', 1): tf.keras.layers.AveragePooling1D,
    ('AveragePooling', 2): tf.keras.layers.AveragePooling2D,
    ('AveragePooling', 3): tf.keras.layers.AveragePooling3D,
    ('Conv', 1): tf.keras.layers.Conv1D,
    ('Conv', 2): tf.keras.layers.Conv2D,
    ('Conv', 3): tf.keras.layers.Conv3D,
    ('Conv', 4): Conv4D,
    ('ConvLSTM', 1): tf.keras.layers.ConvLSTM1D,
    ('ConvLSTM', 2): tf.keras.layers.ConvLSTM2D,
    ('ConvLSTM', 3): tf.keras.layers.ConvLSTM3D,
    ('ConvTranspose', 1): tf.keras.layers.Conv1DTranspose,
    ('ConvTranspose', 2): tf.keras.layers.Conv2DTranspose,
    ('ConvTranspose', 3): tf.keras.layers.Conv3DTranspose,
    ('Cropping', 1): tf.keras.layers.Cropping1D,
    ('Cropping', 2): tf.keras.layers.Cropping2D,
    ('Cropping', 3): tf.keras.layers.Cropping3D,
    ('DepthwiseConv', 1): tf.keras.layers.DepthwiseConv1D,
    ('DepthwiseConv', 2): tf.keras.layers.DepthwiseConv2D,
    ('GlobalAveragePooling', 1): tf.keras.layers.GlobalAveragePooling1D,
    ('GlobalAveragePooling', 2): tf.keras.layers.GlobalAveragePooling2D,
    ('GlobalAveragePooling', 3): tf.keras.layers.GlobalAveragePooling3D,
    ('GlobalMaxPool', 1): tf.keras.layers.GlobalMaxPool1D,
    ('GlobalMaxPool', 2): tf.keras.layers.GlobalMaxPool2D,
    ('GlobalMaxPool', 3): tf.keras.layers.GlobalMaxPool3D,
    ('MaxPool', 1): tf.keras.layers.MaxPool1D,
    ('MaxPool', 2): tf.keras.layers.MaxPool2D,
    ('MaxPool', 3): tf.keras.layers.MaxPool3D,
    ('MaxPool', 4): MaxPool4D,
    
    ('SeparableConv', 1): tf.keras.layers.SeparableConv1D,
    ('SeparableConv', 2): tf.keras.layers.SeparableConv2D,
    ('SpatialDropout', 1): tf.keras.layers.SpatialDropout1D,
    ('SpatialDropout', 2): tf.keras.layers.SpatialDropout2D,
    ('SpatialDropout', 3): tf.keras.layers.SpatialDropout3D,
    ('UpSampling', 1): tf.keras.layers.UpSampling1D,
    ('UpSampling', 2): tf.keras.layers.UpSampling2D,
    ('UpSampling', 3): tf.keras.layers.UpSampling3D,
    ('UpSampling', 4): UpSampling4D,
    ('ZeroPadding', 1): tf.keras.layers.ZeroPadding1D,
    ('ZeroPadding', 2): tf.keras.layers.ZeroPadding2D,
    ('ZeroPadding', 3): tf.keras.layers.ZeroPadding3D
}


class ResizeAndConcatenate(tf.keras.layers.Layer):
  """Resizes and concatenates a list of inputs.

  Similar to `tf.keras.layers.Concatenate`, but if the inputs have different
  shapes, they are resized to match the shape of the first input.

  Args:
    axis: Axis along which to concatenate.
  """
  def __init__(self, axis=-1, **kwargs):
    super().__init__(**kwargs)
    self.axis = axis

  def get_config(self):
        config = super().get_config()
        config.update({
            "axis": self.axis,
        })
        return config

  def call(self, inputs):  
    if not isinstance(inputs, (list, tuple)):
      raise ValueError(
          f"Layer {self.__class__.__name__} expects a list of inputs. "
          f"Received: {inputs}")

    rank = inputs[0].shape.rank
    if rank is None:
      raise ValueError(
          f"Layer {self.__class__.__name__} expects inputs with known rank. "
          f"Received: {inputs}")
    if self.axis >= rank or self.axis < -rank:
      raise ValueError(
          f"Layer {self.__class__.__name__} expects `axis` to be in the range "
          f"[-{rank}, {rank}) for an input of rank {rank}. "
          f"Received: {self.axis}")
    # Canonical axis (always positive).
    axis = self.axis % rank

    # Resize inputs.
    shape = tf.tensor_scatter_nd_update(tf.shape(inputs[0]), [[axis]], [-1])
    resized = [resize_with_crop_or_pad(tensor, shape)
               for tensor in inputs[1:]]

    # Set the static shape for each resized tensor.
    for i, tensor in enumerate(resized):
      static_shape = inputs[0].shape.as_list()
      static_shape[axis] = inputs[i + 1].shape.as_list()[axis]
      static_shape = tf.TensorShape(static_shape)
      resized[i] = tf.ensure_shape(tensor, static_shape)
    return tf.concat(inputs[:1] + resized, axis=self.axis)  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter


tf.random.set_seed(489154)

class unet3plus:
   """
   Class for building a U-Net3+ model.
   """

   def __init__(self,
                inputs,
                filters = [32,64,128,256,512],
                rank = 2,
                out_channels = 1,
                kernel_initializer=tf.keras.initializers.HeNormal(seed=0),
                bias_initializer=tf.keras.initializers.Zeros(),
                kernel_regularizer=None,
                bias_regularizer=None,
                add_dropout = False,
                padding = 'same',
                dropout_rate = 0.5,
                kernel_size = 3,
                out_kernel_size = 3,
                pool_size = 2,
                encoder_block_depth = 2,
                decoder_block_depth = 1,
                batch_norm = True,
                activation = 'relu',
                out_activation = None,
                skip_batch_norm = True,
                skip_type = 'encoder',
                CGM = False,
                deep_supervision = True):
       
       """
       Initialise the U-Net3+ model.
       Args:
           inputs: Input tensor.
           filters: List of filter sizes for each UNet level.
           rank: Number of dimensions (2D or 3D).
           out_channels: Number of output channels (for segmentation this shall be the number of distinct masks).
           kernel_initializer: Initialiser for the convolutional layers.
           bias_initializer: Initialiser for the bias terms.
           kernel_regularizer: Regulariser for the convolutional layers.
           bias_regularizer: Regulariser for the bias terms in convolutional layers.
           add_dropout: Whether to add dropout layers.
           padding: Padding type for the convolutional layers.
           dropout_rate: Dropout rate.
           kernel_size: Kernel size for the convolutional layers.
           out_kernel_size: Kernel size for the final convolutional layers of the network.
           pool_size: Pooling size for the max pooling layers. This can be a tuple specifing the max pool size for each dimension of the input, or a single integer specifying the same size for all dimensions.
           encoder_block_depth: Number of convolutional blocks in each level of the encoding arm.
           decoder_block_depth: Number of convolutional blocks in each level of the decoding arm.
           batch_norm: Whether to use batch normalization.
           activation: Activation function for the convolutional layers.
           out_activation: Activation function for the output layer. For binary segmentation this shall be 'sigmoid' or 'softmax'.
           skip_batch_norm: Whether to use batch normalization in the skip connections.
           skip_type: Type of skip connections to use in the model ('encoder', 'decoder', or 'standard_unet').
           CGM: Whether to use CGM in the model for segmentation (Classification Guided Module).
           deep_supervision: Whether to use deep supervision.
        """
        # Assign parameters
       self.inputs = inputs
       self.filters = filters
       self.levels = len(filters)
       self.rank = rank
       self.out_channels = out_channels
       self.encoder_block_depth = encoder_block_depth
       self.decoder_block_depth = decoder_block_depth
       self.kernel_size = kernel_size
       self.add_dropout = add_dropout
       self.dropout_rate = dropout_rate
       self.skip_type = skip_type  
       self.skip_batch_norm = skip_batch_norm
       self.batch_norm = batch_norm
       self.activation = activation
       self.out_activation = out_activation
       self.CGM = CGM
       self.deep_supervision = deep_supervision
       # Assign pool size based on given tuple, or if single integer is provided, assign the same value to all dimensions using the rank as a guide for the number of dimensions
       if isinstance(pool_size,tuple):
           self.pool_size = pool_size
       else:
           self.pool_size = tuple([pool_size for _ in range(rank)])
        # Assign kernel sizes based on given tuple, or if single integer is provided, assign the same value to all dimensions using the rank as a guide for the number of dimensions
       if isinstance(kernel_size,tuple):
           self.kernel_size = kernel_size
       else:
           self.kernel_size = tuple([kernel_size for _ in range(rank)])
       if isinstance(out_kernel_size,tuple):
           self.out_kernel_size = out_kernel_size
       else:
           self.out_kernel_size = tuple([out_kernel_size for _ in range(rank)])
       # Create the conv and out conv config dictionaries for the conv and out conv layers
       self.conv_config = dict(kernel_size = self.kernel_size,
                          padding = padding,
                          kernel_initializer = kernel_initializer,
                          bias_initializer = bias_initializer,
                          kernel_regularizer = kernel_regularizer,
                          bias_regularizer = bias_regularizer)
       self.out_conv_config = dict(kernel_size = out_kernel_size,
                          padding = padding,
                          kernel_initializer = kernel_initializer,
                          bias_initializer = bias_initializer,
                          kernel_regularizer = kernel_regularizer,
                          bias_regularizer = bias_regularizer)
   
   def aggregate_and_decode(self, input_list, level):
    """
    Aggregates the inputs for the decoder levels and applies convolution to get the output of the decoder level.

    Args:
        input_list: List of inputs to the decoder to be aggregated.
        level: Current decoder level.
    """
    X = ResizeAndConcatenate(name = f'D{level}_input', axis = -1)(input_list) # Takes the various inputs to a decoder level, resizes them to the 1st input size in the list and the concatenates them all.
    X = self.conv_block(X, self.filters[level-1], block_depth = self.decoder_block_depth, conv_block_purpose = 'Decoder', level=level) # Performs a decoder block convolution of the concatenated input (i.e. the concatenated list of filters)
    return X
   
   def deep_sup(self, inputs, level):
    """
    If deep supervision is used, then the network will output a prediction at each level of the decoder.
    This function upsamples the output of a decoder level, convolves it and then applies the output activation function (i.e. to get to the final output).
    If deep supervision is not used, then the network will only output a prediction at the final level of the decoder.

    Args:
        inputs: Input tensor.
        level: Current decoder level.
    """
    conv = get_nd_layer('Conv', self.rank) # gets a convolutional layer of the specified rank (2D or 3D)
    upsamp = get_nd_layer('UpSampling', self.rank) # gets an upsampling layer of the specified rank (2D or 3D)
    size = tuple(np.array(self.pool_size)** (abs(level-1))) # This specifies the amount of upsampling needed to get to the correct final output size. It is the maxpool size to the power of the current decoder level minus one.
    if self.rank == 2:
        upsamp_config = dict(size=size, interpolation='bilinear') # use bilinear interpolation for 2D upsampling
    else:
        upsamp_config = dict(size=size) # for 3D upsampling, you cannot do bilinear interpolation, so this just uses the default upsampling method.
    X = inputs  
    X = conv(self.out_channels, activation = None, **self.out_conv_config, name = f'deepsup_conv_{level}_1')(X) # Convolves the input to get the correct number of output channels
    if level != 1:
        X = upsamp(**upsamp_config, name = f'deepsup_upsamp_{level}')(X) # Upsamples the convolved input to the correct size for the final output
    X = conv(self.out_channels, activation = None, **self.out_conv_config, name = f'deepsup_conv_{level}_2')(X) # Convolves the upsampled input to get the correct number of output channels (e.g. to correct artifacts due to upsampling)
    if self.out_activation:
        X = tf.keras.layers.Activation(activation = self.out_activation, name = f'Output_{level}', dtype='float32')(X) # Applies the output activation function to get the final output
    return X
       
       
       
   def skip_connection(self, inputs, to_level, from_level):
    """
    This function takes an input tensor and processes it as a skip connection to the decoder level.

    Args:
        inputs: Input tensor.
        to_level: Current decoder level.
        from_level: Level of UNet the input tensor is from.    
    """
    conv = get_nd_layer('Conv', self.rank) # gets a convolutional layer of the specified rank (2D or 3D)
    level_diff = from_level - to_level  # difference between level of decoder and level of UNet the input tensor is from
    size = tuple(np.array(self.pool_size)** (abs(level_diff))) # This specifies the amount of upsampling needed to get to the correct size for decoder level. It is the maxpool size to the power of the level difference.
    maxpool = get_nd_layer('MaxPool', self.rank) # gets a maxpool layer of the specified rank (2D or 3D)
    upsamp = get_nd_layer('UpSampling', self.rank) # gets an upsampling layer of the specified rank (2D or 3D)
    if self.rank == 2:
        upsamp_config = dict(size=size, interpolation='bilinear') # use bilinear interpolation for 2D upsampling
    else:
        upsamp_config = dict(size=size) # for 3D upsampling, you cannot do bilinear interpolation, so this just uses the default upsampling method.
    
    X = inputs        
    if to_level < from_level: # If coming from a deeper level of the UNet, then we need to upsample the input tensor to the correct size for the decoder level
        X = upsamp(**upsamp_config, name = f'Skip_Upsample_{from_level}_{to_level}')(X)
    elif to_level > from_level: # If coming from a shallower level of the UNet, then we need to maxpool the input tensor to the correct size for the decoder level
        X = maxpool(pool_size = size, name = f'Skip_Maxpool_{from_level}_{to_level}')(X)
    
    if self.skip_batch_norm: # If using batch normalization in the skip connections, then apply it within the conv block
        X = self.conv_block(X, self.filters[to_level-1], block_depth = self.decoder_block_depth, conv_block_purpose ='Skip', level = f'{from_level}_{to_level}') # applies conv block to the upsampled/maxpooled input tensor (with batch normalization)
    else:
        X = conv(self.filters[to_level-1],**self.conv_config, name = f'Skip_Conv_{from_level}_{to_level}')(X)  # applies conv layer to the upsampled/maxpooled input tensor (without batch normalization)
        
    return X # note: returns the output of a single skip connection, but does not yet concatenate the output to the other skip outputs or existing decoder level filters. 
       
   def conv_block(self, inputs, filters, block_depth, conv_block_purpose, level):
       """
       This function creates a convolutional block with the specified number of stacks and filters.
         Args:
                inputs: Input tensor.
                filters: Number of filters for the convolutional layers.
                block_depth: Number of convolutional stacks in the block.
                conv_block_purpose: Type of conv block (Encoder, Decoder, Skip).
                level: Current level level.
       """
       conv = get_nd_layer('Conv', self.rank) # gets a convolutional layer of the specified rank (2D or 3D)
       X = inputs
       for i in range(block_depth): # replicate the conv block, depth number of times
           X = conv(filters, **self.conv_config, name = f'{conv_block_purpose}{level}_Conv_{i+1}')(X) # applies conv layer to the input tensor
           if self.batch_norm: # If using batch normalization, then apply it after the conv layer
               X = tf.keras.layers.BatchNormalization(axis=-1, name = f'{conv_block_purpose}{level}_BN_{i+1}')(X) 
           if self.activation: # If using an activation function, then apply it after the conv layer
            X = tf.keras.layers.Activation(activation = self.activation, name = f'{conv_block_purpose}{level}_Activation_{i+1}')(X)
       return X
   
   
   def encode(self, inputs, level, block_depth):
       """
       Creates the encoding block of the U-Net3+ model.

         Args:
                inputs: Input tensor.
                level: Current level level.
                block_depth: Number of convolutional stacks in the block.
       """
       maxpool = get_nd_layer('MaxPool', self.rank) # gets a maxpool layer of the specified rank (2D or 3D)
       level -= 1 # python indexing
       filters = self.filters[level] # get the number of filters for the current level
       X = inputs
       if level != 0: # 0 is the input level, so we do not need to maxpool it
           X = maxpool(pool_size=self.pool_size, name = f'encoding_{level}_maxpool')(X) # maxpool the input tensor to the correct size for the next level
       X = self.conv_block(X, filters, block_depth, conv_block_purpose = 'Encoder', level = level+1) # applies conv block to the maxpooled input tensor
       if level == (self.levels-1) and self.add_dropout: # Check if level is the bottom level of the UNet, and if so, apply dropout if specified
           X = tf.keras.layers.Dropout(rate = self.dropout_rate, name = f'Encoder{level+1}_dropout')(X)
       return X
       
   def outputs(self):
       """
       This is the build function for the U-Net3+ model. 

       """
       XE  = [self.inputs] # This is a list of encoder level outputs, starting with the input tensor
       for i in range(self.levels): # for each level of the UNet, we apply an encoding block to the output of the previous level
           XE.append(self.encode(XE[i], level = i+1, block_depth = self.encoder_block_depth))
       XD = [XE[-1]] # This is a list of decoder level outputs, starting with the output of the last encoder level
       if self.skip_type == 'encoder': 
           # If using encoder-type skip connections, then we apply skip connections from every encoder level to the current decoder level - except the encoder level one deeper. For this level, we use the output of the last decoder level.
           for decoder_level in range(self.levels-1,0,-1): # build the decoder levels in reverse order
               input_contributions = []
               for unet_level in range(1,self.levels+1):
                   if unet_level == decoder_level+1: # If the unet level is one deeper than the decoder level, then we get a skip connection from the output of the last decoder level
                       input_contributions.append(self.skip_connection(XD[-1], decoder_level, unet_level))
                   else: # Otherwise we get a skip connection from the output of the encoder level
                       input_contributions.append(self.skip_connection(XE[unet_level], decoder_level, unet_level))
               XD.append(self.aggregate_and_decode(input_contributions,decoder_level)) # aggregate and conv the skip connections to the current decoder level. This gives the output of the decoder level. Append this to the list of decoder level outputs.
       elif self.skip_type == 'decoder':
           # If using decoder-type skip connections, then 
           for decoder_level in range(self.levels-1,0,-1):
               skip_contributions = []
               # Append skips from encoder
               for encoder_level in range(1,decoder_level+1): # All encoders shallower or equal to the decoder level contribute a skip connection
                   skip_contributions.append(self.skip_connection(XE[encoder_level], decoder_level, encoder_level))
               # Append skips from decoder
               for i in range(len(XD)-1,-1,-1): # All decoders deeper than the current decoder level contribute a skip connection (note: XD is build iteratively in a loop from the deepest level upwards. Therefore at each stage of the loop, XD grows and deeper decoder levels contribute skip connections to the current decoder level)
                   skip_contributions.append(self.skip_connection(XD[i], decoder_level, (self.levels-i)))
               XD.append(self.aggregate_and_decode(skip_contributions,decoder_level)) # aggregate and conv the skip connections to the current decoder level. This gives the output of the decoder level. Append this to the list of decoder level outputs.
       elif self.skip_type == 'standard_unet':
           # If standard_unet type skips, then at each decoder level, we get a skip connection from the corresponding encoder level 
           for decoder_level in range(self.levels-1,0,-1): 
               skip_contributions = [XE[decoder_level],self.skip_connection(XD[-1],decoder_level,decoder_level+1)]
               XD.append(self.aggregate_and_decode(skip_contributions,decoder_level)) # aggregate and conv the skip connections to the current decoder level.
       else:
           raise ValueError(f"Invalid skip_type")
       if self.deep_supervision == True:
           XD = [self.deep_sup(xd, self.levels-i) for i,xd in enumerate(XD)] # If deep supervision is used, then we apply deep supervision to each decoder level output
           return XD
       else:
           XD[-1] = self.deep_sup(XD[-1],1) # If deep supervision is not used, then we only apply deep supervision to the final decoder level output
           return XD[-1]
       
       
def build_unet3plus_4D(input_shape, num_classes):

    input_shape =  [32, None, 128, 128, 1] # T, D, H, W, C
    output_shape = [32, None, 128, 128, num_classes]

    inputs = tf.keras.Input(shape = input_shape)
    unet3 = unet3plus(inputs, 
                    filters =  [25 * 2**i for i in range(4)],
                    rank = 4,
                    kernel_size= (3,3,3),
                    out_kernel_size = (1,1,1),
                    out_channels = num_classes,
                    kernel_initializer=tf.keras.initializers.HeNormal(seed=0),
                    pool_size = (1,1,2,2),
                    batch_norm = True,
                    activation = 'LeakyReLU',
                    add_dropout=False,
                    dropout_rate=0.3,
                    out_activation = 'softmax',
                    skip_type = 'encoder',
                    deep_supervision=True) 

    model = tf.keras.Model(inputs = inputs, outputs = unet3.outputs())
    return model