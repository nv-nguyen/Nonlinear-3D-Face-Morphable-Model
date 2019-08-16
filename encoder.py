from __future__ import division, print_function, absolute_import
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


from layer_creator import add_convolutive_layer, add_full_connected_layer,add_average_pooling_layer
m = 6 # dim of projection parameter
dim_encoding_shape=160 # dim of shape base
dim_encoding_texture=160 


#Use initialisation of Xavier for weights
initializer_xavier = tf.contrib.layers.xavier_initializer_conv2d()
# Store layers conv/weight & bias
weights_encoder = {
    # 3x3 conv, 3 input, 32 outputs
    "conv11/weights_encoder11": tf.get_variable("conv11/weights_encoder11", shape = [3, 3, 3, 32] , initializer=initializer_xavier ),
    # 3x3 conv, 32 inputs, 64 outputs
    "conv12/weights_encoder12": tf.get_variable("conv12/weights_encoder12", shape = [3, 3, 32, 64] ,  initializer = initializer_xavier ),
    #--------------------------------------------------------
    # 3x3 conv, 64 inputs, 64 outputs
    "conv21/weights_encoder21": tf.get_variable("conv21/weights_encoder21", shape = [3, 3, 64, 64] ,  initializer = initializer_xavier),
    # 3x3 conv, 64 inputs, 64 outputs
    "conv22/weights_encoder22": tf.get_variable("conv22/weights_encoder22", shape = [3, 3, 64, 64] ,  initializer = initializer_xavier),
    # 3x3 conv, 64 inputs, 128 outputs
    "conv23/weights_encoder23": tf.get_variable("conv23/weights_encoder23", shape = [3, 3, 64, 128] ,  initializer = initializer_xavier),
    #--------------------------------------------------------
    # 3x3 conv, 128 inputs, 128 outputs
    "conv31/weights_encoder31": tf.get_variable("conv31/weights_encoder31", shape = [3, 3, 128, 128] ,  initializer = initializer_xavier),
    # 3x3 conv, 128 inputs, 96 outputs 
    "conv32/weights_encoder32": tf.get_variable("conv32/weights_encoder32", shape = [3, 3, 128, 96] ,  initializer = initializer_xavier),
    # 3x3 conv, 96 inputs, 192 outputs
    "conv33/weights_encoder33": tf.get_variable("conv33/weights_encoder33", shape = [3, 3, 96, 192] ,  initializer = initializer_xavier),
    #-------------------------------------------------------
    # 3x3 conv, 192 inputs, 192 outputs
    "conv41/weights_encoder41": tf.get_variable("conv41/weights_encoder41", shape = [3, 3, 192, 192] ,  initializer = initializer_xavier),
    # 3x3 conv, 192 inputs, 128 outputs
    "conv42/weights_encoder42": tf.get_variable("conv42/weights_encoder42", shape = [3, 3, 192, 128] ,  initializer = initializer_xavier),
    # 3x3 conv, 128 inputs, 256 outputs
    "conv43/weights_encoder43": tf.get_variable("conv43/weights_encoder43", shape = [3, 3, 128, 256] ,  initializer = initializer_xavier),
    #------------------------------------------------------
    # 3x3 conv, 256 inputs, 256 outputs
    "conv51/weights_encoder51": tf.get_variable("conv51/weights_encoder51", shape = [3, 3, 256, 256] ,  initializer = initializer_xavier),
    # 3x3 conv, 256 inputs, 160 outputs
    "conv52/weights_encoder52": tf.get_variable("conv52/weights_encoder52", shape = [3, 3, 256, 160] ,  initializer = initializer_xavier),
    # 3x3 conv, 160 inputs, dim_encoding_shape+dim_encoding_texture+64 outputs
    "conv53/weights_encoder53": tf.get_variable("conv53/weights_encoder53", shape = [3, 3, 160, dim_encoding_shape+dim_encoding_texture+64] ,  
                                                                                initializer = initializer_xavier),
    # fully connected, 64 inputs, m outputs
    "Encoder/weight_full_connected_layer": tf.get_variable("Encoder/weight_full_connected_layer", shape = [64, m], initializer = initializer_xavier)
}

#Initialisation to zero for all bias 
initializer_zero = tf.zeros_initializer
bias_encoder = {
    "conv11/bias_encoder11": tf.get_variable("conv11/bias_encoder11", shape = [32] , initializer = initializer_zero),
    "conv12/bias_encoder12": tf.get_variable("conv12/bias_encoder12", shape = [64] , initializer = initializer_zero), 
    #--------------------------------------------
    "conv21/bias_encoder21": tf.get_variable("conv21/bias_encoder21", shape = [64] , initializer = initializer_zero),
    "conv22/bias_encoder22": tf.get_variable("conv22/bias_encoder22", shape = [64] , initializer =initializer_zero),
    "conv23/bias_encoder23": tf.get_variable("conv23/bias_encoder23", shape = [128] , initializer = initializer_zero),
    #----------------------------------------------
    "conv31/bias_encoder31": tf.get_variable("conv31/bias_encoder31", shape = [128] , initializer = initializer_zero),
    "conv32/bias_encoder32": tf.get_variable("conv32/bias_encoder32", shape = [96] , initializer = initializer_zero),
    "conv33/bias_encoder33": tf.get_variable("conv33/bias_encoder33", shape = [192] , initializer = initializer_zero),
    #----------------------------------------------
    "conv41/bias_encoder41": tf.get_variable("conv41/bias_encoder41", shape = [192] , initializer = initializer_zero),
    "conv42/bias_encoder42": tf.get_variable("conv42/bias_encoder42", shape = [128] , initializer = initializer_zero),
    "conv43/bias_encoder43": tf.get_variable("conv43/bias_encoder43", shape = [256] , initializer = initializer_zero),
    #----------------------------------------------
    "conv51/bias_encoder51": tf.get_variable("conv51/bias_encoder51", shape = [256] , initializer = initializer_zero),
    "conv52/bias_encoder52": tf.get_variable("conv52/bias_encoder52", shape = [160] , initializer = initializer_zero),
    "conv53/bias_encoder53": tf.get_variable("conv53/bias_encoder53", shape = [dim_encoding_shape+dim_encoding_texture+64] , 
                                                                        initializer = initializer_zero),
    "Encoder/bias_full_connected_layer": tf.get_variable("Encoder/bias_full_connected_layer", shape = [6] , initializer = initializer_zero)
}

# Create model
def encoder (image):
#def encoder(x, weights_encoder, bias_encoder, dropout):
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
        
        conv11 = tf.nn.elu (add_convolutive_layer(image, weights_encoder["conv11/weights_encoder11"], bias_encoder["conv11/bias_encoder11"]))

        # Convolution Layer 12
        conv12 = tf.nn.elu (add_convolutive_layer(conv11, weights_encoder["conv12/weights_encoder12"], bias_encoder["conv12/bias_encoder12"]))

        #-------------------------------------------------------------------------
        # Convolution Layer 21
        conv21 = tf.nn.elu (add_convolutive_layer(conv12, weights_encoder["conv21/weights_encoder21"], bias_encoder["conv21/bias_encoder21"],stride =2))

        # Convolution Layer 22
        conv22 = tf.nn.elu (add_convolutive_layer(conv21, weights_encoder["conv22/weights_encoder22"], bias_encoder["conv22/bias_encoder22"]))

        # Convolution Layer 23
        conv23 = tf.nn.elu (add_convolutive_layer(conv22, weights_encoder["conv23/weights_encoder23"], bias_encoder["conv23/bias_encoder23"]))
        #-------------------------------------------------------------------------

        # Convolution Layer 31
        conv31 = tf.nn.elu (add_convolutive_layer(conv23, weights_encoder["conv31/weights_encoder31"], bias_encoder["conv31/bias_encoder31"],stride =2))

        # Convolution Layer 32
        conv32 = tf.nn.elu (add_convolutive_layer(conv31, weights_encoder["conv32/weights_encoder32"], bias_encoder["conv32/bias_encoder32"]))

        # Convolution Layer 33
        conv33 = tf.nn.elu (add_convolutive_layer(conv32, weights_encoder["conv33/weights_encoder33"], bias_encoder["conv33/bias_encoder33"]))

        #-------------------------------------------------------------------------

        # Convolution Layer 41
        conv41 = tf.nn.elu (add_convolutive_layer(conv33, weights_encoder["conv41/weights_encoder41"], bias_encoder["conv41/bias_encoder41"],stride =2))

        # Convolution Layer 42
        conv42 = tf.nn.elu (add_convolutive_layer(conv41, weights_encoder["conv42/weights_encoder42"], bias_encoder["conv42/bias_encoder42"]))

        # Convolution Layer 43
        conv43 = tf.nn.elu (add_convolutive_layer(conv42, weights_encoder["conv43/weights_encoder43"], bias_encoder["conv43/bias_encoder43"]))

        #-------------------------------------------------------------------------


        # Convolution Layer 51
        conv51 = tf.nn.elu (add_convolutive_layer(conv43, weights_encoder["conv51/weights_encoder51"], bias_encoder["conv51/bias_encoder51"],stride =2))

        # Convolution Layer 52
        conv52 = tf.nn.elu (add_convolutive_layer(conv51, weights_encoder["conv52/weights_encoder52"], bias_encoder["conv52/bias_encoder52"]))
 
        # Convolution Layer 53
        conv53 = tf.nn.conv2d(conv52, weights_encoder["conv53/weights_encoder53"], strides=[1, 1, 1, 1], padding='SAME')
        conv53_bias =tf.nn.elu ( tf.nn.bias_add(conv53, bias_encoder["conv53/bias_encoder53"]))    

        #--------------------------------------------------------------------------

        #Average_pooling layer
        avgpool = add_average_pooling_layer(conv53_bias)

        #VN :  convert from 1x1x64 to 64 (it means delete 1x1)
        avgpool_2d = tf.squeeze(avgpool, [1, 2])
        #seperate into 3 parts( dim_encoding_shape  dim_encoding_texture, one part for m)
        shapeEncoding= avgpool_2d[:,:dim_encoding_shape]
        
        textureEncoding= avgpool_2d[:,dim_encoding_shape:dim_encoding_shape+dim_encoding_texture]
        
        cameraSetupPreLayer = avgpool_2d[:,dim_encoding_shape+dim_encoding_texture: ]

        # Fully connected layer
        cameraSetup = add_full_connected_layer (cameraSetupPreLayer , weights_encoder["Encoder/weight_full_connected_layer"], 
                                                                          bias_encoder["Encoder/bias_full_connected_layer"])

    return shapeEncoding, textureEncoding, cameraSetup