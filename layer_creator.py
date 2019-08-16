import tensorflow as tf
from functools import partial
# Create some functions with defaut parameters 

conv           = partial( tf.nn.conv2d                     , 
                          padding     = 'SAME'             ,  
                        )

conv_transpose = partial(  tf.nn.conv2d_transpose             , 
                           padding     =  'SAME'              ,  
                        )

training = tf.placeholder_with_default(True, shape = (), name = 'training')

batch_norm     = partial( tf.layers.batch_normalization    , 
                          training =  training             , 
                          momentum =  0.9                  , 
                        )

# Function activation is not used by default in full connected layer
def add_full_connected_layer(inputs, weights, bias): 
    full_connected_layer = tf.nn.bias_add(tf.matmul(inputs, weights), bias)
    return full_connected_layer

def add_full_connected_normalized_layer(inputs, weights, bias): 
    full_connected_layer = tf.nn.bias_add(tf.matmul(inputs, weights), bias)
    normalized = batch_norm(full_connected_layer)
    return normalized
    
def add_convolutive_layer(inputs, weights, bias, stride = 1):
    convolutive_layer = conv(inputs, weights, strides =[1, stride, stride, 1])
    convolutive_layer_bias = tf.nn.bias_add(convolutive_layer, bias)
    # normalized = batch_norm(convolutive_layer_bias)
    # return activation(normalized)
    return convolutive_layer_bias
def add_convolutive_transposed_layer(inputs, weights, shape_output, bias, stride = 2):
    conv_transpose_layer = conv_transpose(inputs, weights, output_shape = shape_output,  strides =[1, stride, stride, 1])
    conv_transpose_layer_bias =  tf.nn.bias_add(conv_transpose_layer, bias)
    # normalized = batch_norm(conv_transpose_layer_bias)
    # return activation(normalized)
    return conv_transpose_layer_bias

def add_average_pooling_layer(x, size_window = 6):
    return tf.nn.avg_pool(x, ksize=[1, size_window, size_window, 1], strides=[1, size_window, size_window, 1],
                          padding='SAME')
#--------------------------------------------------


