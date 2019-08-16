from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from layer_creator import add_convolutive_layer, add_convolutive_transposed_layer,add_full_connected_layer, add_full_connected_normalized_layer
dim_encoding_shape = 160
dim_encoding_texture = 160
VERTEX_NUM = 5904
# TRI_NUM = 11438


initializer_xavier = tf.contrib.layers.xavier_initializer_conv2d()

# Store layers weight & bias
weights_decoder_texture = {
    #FC layer
    "Decoder/weights_full_connected_layer" : tf.get_variable( "Decoder/weights_full_connected_layer", shape = [dim_encoding_texture, 6*6*320], 
                                                                                                                initializer=initializer_xavier),
    # 3x3 conv, 320 input, 160 decoder_shape_outputs
    "FConv52/weights_decoder_texture52": tf.get_variable( "FConv52/weights_decoder_texture52" ,shape = [3, 3, 320, 160], initializer=initializer_xavier),
    # 3x3 conv, 160 inputs, 256 decoder_shape_outputs
    "FConv51/weights_decoder_texture51": tf.get_variable( "FConv51/weights_decoder_texture51",shape = [3, 3, 160, 256], initializer=initializer_xavier),
    #--------------------------------------------------------
    # 3x3 conv, 256 inputs, 256 decoder_shape_outputs
    "FConv43/weights_decoder_texture43": tf.get_variable( "FConv43/weights_decoder_texture43",shape = [3, 3, 256, 256], initializer=initializer_xavier),
    # 3x3 conv, 256 inputs, 128 decoder_shape_outputs
    "FConv42/weights_decoder_texture42": tf.get_variable( "FConv42/weights_decoder_texture42",shape = [3, 3, 256, 128], initializer=initializer_xavier),
    # 3x3 conv, 128 inputs, 192 decoder_shape_outputs
    "FConv41/weights_decoder_texture41": tf.get_variable( "FConv41/weights_decoder_texture41",shape = [3, 3, 128, 192], initializer=initializer_xavier),
    #--------------------------------------------------------
    # 3x3 conv, 192 inputs, 192 decoder_shape_outputs
    "FConv33/weights_decoder_texture33": tf.get_variable( "FConv33/weights_decoder_texture33",shape = [3, 3, 192, 192], initializer=initializer_xavier),
    # 3x3 conv, 192 inputs, 96 decoder_shape_outputs
    "FConv32/weights_decoder_texture32": tf.get_variable( "FConv32/weights_decoder_texture32",shape = [3, 3, 192, 96], initializer=initializer_xavier),
    # 3x3 conv, 96 inputs, 128 decoder_shape_outputs
    "FConv31/weights_decoder_texture31": tf.get_variable( "FConv31/weights_decoder_texture31",shape = [3, 3, 96, 128], initializer=initializer_xavier),
    #-------------------------------------------------------
    # 3x3 conv, 128 inputs, 128 decoder_shape_outputs
    "FConv23/weights_decoder_texture23": tf.get_variable( "FConv23/weights_decoder_texture23",shape = [3, 3, 128, 128], initializer=initializer_xavier),
    # 3x3 conv, 128 inputs, 64 decoder_shape_outputs
    "FConv22/weights_decoder_texture22": tf.get_variable( "FConv22/weights_decoder_texture22",shape = [3, 3, 128, 64], initializer=initializer_xavier),
    # 3x3 conv, 64 inputs, 64 decoder_shape_outputs
    "FConv21/weights_decoder_texture21": tf.get_variable( "FConv21/weights_decoder_texture21",shape = [3, 3, 64, 64], initializer=initializer_xavier),
    #------------------------------------------------------
    # 3x3 conv, 64 inputs, 64 decoder_shape_outputs
    "FConv13/weights_decoder_texture13": tf.get_variable( "FConv13/weights_decoder_texture13",shape = [3, 3, 64, 64], initializer=initializer_xavier),
    # 3x3 conv, 64 inputs, 32 decoder_shape_outputs
    "FConv12/weights_decoder_texture12": tf.get_variable( "FConv12/weights_decoder_texture12",shape = [3, 3, 64, 32], initializer=initializer_xavier),
    # 3x3 conv, 160 inputs, dim_encoding_shape+dim_encoding_texture+64 decoder_shape_outputs
    "FConv11/weights_decoder_texture11": tf.get_variable( "FConv11/weights_decoder_texture11",shape = [3, 3, 32, 3], initializer=initializer_xavier)
    
}
initializer_zero = tf.zeros_initializer

bias_decoder_texture = {
    #FC layer
    "Decoder/bias_full_connected_layer" : tf.get_variable( "Decoder/bias_full_connected_layer",shape = [320], initializer = initializer_zero),
    #-------------------------------------------
    "FConv52/bias_decoder_texture52": tf.get_variable( "FConv52/bias_decoder_texture52",shape = [160], initializer = initializer_zero),
    "FConv51/bias_decoder_texture51": tf.get_variable( "FConv51/bias_decoder_texture51",shape = [256], initializer = initializer_zero),
    #--------------------------------------------
    "FConv43/bias_decoder_texture43": tf.get_variable( "FConv43/bias_decoder_texture43",shape = [256], initializer = initializer_zero),
    "FConv42/bias_decoder_texture42": tf.get_variable( "FConv42/bias_decoder_texture42",shape = [128], initializer = initializer_zero),
    "FConv41/bias_decoder_texture41": tf.get_variable( "FConv41/bias_decoder_texture41",shape = [192], initializer = initializer_zero),
    #----------------------------------------------
    "FConv33/bias_decoder_texture33": tf.get_variable( "FConv33/bias_decoder_texture33",shape = [192], initializer = initializer_zero),
    "FConv32/bias_decoder_texture32": tf.get_variable( "FConv32/bias_decoder_texture32",shape = [96], initializer = initializer_zero),
    "FConv31/bias_decoder_texture31": tf.get_variable( "FConv31/bias_decoder_texture31",shape = [128], initializer = initializer_zero),
    #----------------------------------------------
    "FConv23/bias_decoder_texture23": tf.get_variable( "FConv23/bias_decoder_texture23",shape = [128], initializer = initializer_zero),
    "FConv22/bias_decoder_texture22": tf.get_variable( "FConv22/bias_decoder_texture22",shape = [64], initializer = initializer_zero),
    "FConv21/bias_decoder_texture21": tf.get_variable( "FConv21/bias_decoder_texture21",shape = [64], initializer = initializer_zero),
    #----------------------------------------------
    "FConv13/bias_decoder_texture13": tf.get_variable( "FConv13/bias_decoder_texture13",shape = [64], initializer = initializer_zero),
    "FConv12/bias_decoder_texture12": tf.get_variable( "FConv12/bias_decoder_texture12",shape = [32], initializer = initializer_zero),
    "FConv11/bias_decoder_texture11": tf.get_variable( "FConv11/bias_decoder_texture11",shape = [3], initializer = initializer_zero)
    #----------------------------------------------
}

#--------------------------------------------------------------------------------------------------------------

#Decoder a tensor with shape [None,dim_encoding_texture] into  [None,128,128,3]
def decoder_texture(textureEncoding):
    with tf.variable_scope("decoder_texture", reuse=tf.AUTO_REUSE):
        # Tensor input become 2-D: [Batch Size,dim_encoding_texture]
        #Full connectd layer 
        full_connected_layer_2d = tf.matmul(textureEncoding, weights_decoder_texture["Decoder/weights_full_connected_layer"])

        #Convert tensor 2D [Batch Size,6*6*320] to tensor 4D [Batch Size,6,6,320]
        batch_size = tf.shape(textureEncoding)[0]
        full_connected_layer_4d = tf.reshape(full_connected_layer_2d, [batch_size,6,6,320])
        full_connected_layer_4d_added_bias = tf.nn.bias_add(full_connected_layer_4d, bias_decoder_texture["Decoder/bias_full_connected_layer"])
        
        full_connected_layer = tf.cast(full_connected_layer_4d_added_bias, tf.float32)
        
        # Convolution Layer 52
        conv52 = tf.nn.elu( add_convolutive_layer(full_connected_layer, weights_decoder_texture["FConv52/weights_decoder_texture52"], bias_decoder_texture["FConv52/bias_decoder_texture52"]))
        
        # Add padding for FConv52 layer, so output size is converted from 6x6x160 to 8x8x160
        paddings = tf.constant ([[0,0],[1,1],[1,1],[0,0]], dtype = tf.int32)
        conv52_paddings = tf.pad(conv52,paddings, "CONSTANT")
        
        # Convolution Layer 51
        conv51 = tf.nn.elu( add_convolutive_layer(conv52_paddings, weights_decoder_texture["FConv51/weights_decoder_texture51"], bias_decoder_texture["FConv51/bias_decoder_texture51"]))
        
        #-------------------------------------------------------------------------
        # Convolution Layer 43
        conv43 = tf.nn.elu( add_convolutive_transposed_layer(conv51, weights_decoder_texture["FConv43/weights_decoder_texture43"],  [batch_size, 16, 16, 256], bias_decoder_texture["FConv43/bias_decoder_texture43"]))

        # Convolution Layer 42
        conv42 = tf.nn.elu( add_convolutive_layer(conv43, weights_decoder_texture["FConv42/weights_decoder_texture42"],   bias_decoder_texture["FConv42/bias_decoder_texture42"]))

        # Convolution Layer 41
        conv41 = tf.nn.elu( add_convolutive_layer(conv42, weights_decoder_texture["FConv41/weights_decoder_texture41"],   bias_decoder_texture["FConv41/bias_decoder_texture41"]))
        #-------------------------------------------------------------------------

        # Convolution Layer 33
        conv33 = tf.nn.elu( add_convolutive_transposed_layer(conv41, weights_decoder_texture["FConv33/weights_decoder_texture33"],  [batch_size, 32, 32, 192], bias_decoder_texture["FConv33/bias_decoder_texture33"]))

        # Convolution Layer 32
        conv32 = tf.nn.elu( add_convolutive_layer(conv33, weights_decoder_texture["FConv32/weights_decoder_texture32"],   bias_decoder_texture["FConv32/bias_decoder_texture32"]))

        # Convolution Layer 31
        conv31 = tf.nn.elu( add_convolutive_layer(conv32, weights_decoder_texture["FConv31/weights_decoder_texture31"],  bias_decoder_texture["FConv31/bias_decoder_texture31"]))

        #-------------------------------------------------------------------------

        # Convolution Layer 23
        conv23 = tf.nn.elu( add_convolutive_transposed_layer(conv31, weights_decoder_texture["FConv23/weights_decoder_texture23"],  [batch_size, 64, 64, 128], bias_decoder_texture["FConv23/bias_decoder_texture23"]))

        # Convolution Layer 22
        conv22 = tf.nn.elu( add_convolutive_layer(conv23, weights_decoder_texture["FConv22/weights_decoder_texture22"],   bias_decoder_texture["FConv22/bias_decoder_texture22"]))

        # Convolution Layer 21
        conv21 = tf.nn.elu( add_convolutive_layer(conv22, weights_decoder_texture["FConv21/weights_decoder_texture21"],   bias_decoder_texture["FConv21/bias_decoder_texture21"]))

        #-------------------------------------------------------------------------


        # Convolution Layer 13
        conv13 = tf.nn.elu( add_convolutive_transposed_layer(conv21, weights_decoder_texture["FConv13/weights_decoder_texture13"],  [batch_size, 128, 128, 64], bias_decoder_texture["FConv13/bias_decoder_texture13"]))

        # Convolution Layer 12
        conv12 = tf.nn.elu( add_convolutive_layer(conv13, weights_decoder_texture["FConv12/weights_decoder_texture12"],   bias_decoder_texture["FConv12/bias_decoder_texture12"]))

        # Convolution Layer 11
        texture = add_convolutive_layer(conv12, weights_decoder_texture["FConv11/weights_decoder_texture11"],bias_decoder_texture["FConv11/bias_decoder_texture11"])

    return tf.nn.tanh(texture)




num_neuron_of_hidden_layer = 1000
weights_decoder_shape = {
    #layer 1
    "decoder_shape_layer1/weights_decoder_shape1": tf.get_variable( "decoder_shape_layer1/weights_decoder_shape1", shape = [dim_encoding_shape, num_neuron_of_hidden_layer], initializer=initializer_xavier),
    #layer 2
    "decoder_shape_layer2/weights_decoder_shape2": tf.get_variable( "decoder_shape_layer2/weights_decoder_shape2" , shape = [num_neuron_of_hidden_layer, num_neuron_of_hidden_layer], initializer=initializer_xavier),
    #layer decoder_shape_layer3
    "decoder_shape_layer3/weight_decoder_shape_layer3": tf.get_variable( "decoder_shape_layer3/weight_decoder_shape_layer3" , shape = [num_neuron_of_hidden_layer, 3*VERTEX_NUM], initializer=initializer_xavier)
}

bias_decoder_shape = {
    #layer 1
    "decoder_shape_layer1/bias_decoder_shape1": tf.get_variable( "decoder_shape_layer1/bias_decoder_shape1", shape = [num_neuron_of_hidden_layer], initializer = initializer_zero),
    #layer 2
    "decoder_shape_layer2/bias_decoder_shape2": tf.get_variable( "decoder_shape_layer2/bias_decoder_shape2", shape = [num_neuron_of_hidden_layer], initializer = initializer_zero),
    #layer decoder_shape_layer3 
    "decoder_shape_layer3/bias_decoder_shape_layer3": tf.get_variable( "decoder_shape_layer3/bias_decoder_shape_layer3", shape = [3*VERTEX_NUM], initializer = initializer_zero)
}



def decoder_shape(shapeEncoding):
    with tf.variable_scope("decoder_shape", reuse=tf.AUTO_REUSE):
        #decoder_shape_layer1
        decoder_shape_layer1 = tf.nn.elu(add_full_connected_normalized_layer(shapeEncoding, weights_decoder_shape["decoder_shape_layer1/weights_decoder_shape1"], bias_decoder_shape["decoder_shape_layer1/bias_decoder_shape1"]))
        #decoder_shape_layer2
        decoder_shape_layer2 = tf.nn.elu(add_full_connected_normalized_layer(decoder_shape_layer1, weights_decoder_shape["decoder_shape_layer2/weights_decoder_shape2"], bias_decoder_shape["decoder_shape_layer2/bias_decoder_shape2"]))
        #decoder_shape_layer3
        decoder_shape_layer3 = add_full_connected_layer(decoder_shape_layer2, weights_decoder_shape["decoder_shape_layer3/weight_decoder_shape_layer3"], bias_decoder_shape["decoder_shape_layer3/bias_decoder_shape_layer3"])
        #shape = tf.reshape(decoder_shape_layer3_with_bias, shape = [-1,VERTEX_NUM,3] )
        shape = tf.reshape(decoder_shape_layer3, shape = [1,VERTEX_NUM,3] )    
        return shape

