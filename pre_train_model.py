#In this pre-train model, we need to load the data "facegen_data.npz" and défine the path to save the result.
#By defaut, the data is in the same directory, the result is being saved in "./output/pre_train_model/"
#The weights and bias after pre-train will be saved in './save_model_pre_train/' and will be sent to train_model

from __future__ import division, print_function, absolute_import
import numpy as np
import tensorflow as tf
import os
import sys
from encoder import encoder 
from decoder import decoder_texture, decoder_shape 
from utils_in_out import save_images
from engine_rendering_FaceGen import warp_uvTexture
#----------------------------------------------------------------------------------------
#Input and output by defaut

#import 200 photos and theirs labels S,T,m for pre-train model
data = np.load("facegen_data.npz")
data_shape = data["S"]
data_texture = data["T"]
data_camera = data["m"]
data_image = data["image"]

#save_path ofr image reconstruited
save_result_path = './output/pre_train_model/'
#save_path for pre_train_model
save_pre_train_model_path = './save_model_pre_train/my-test-model.ckpt'
#----------------------------------------------------------------------
_CUDA_DIR = '/CUDA_ZBUFFER_KERNEL/'
dim_encoding_shape=160 # dim of shape base
dim_encoding_texture=160 
m_dim = 6
VERTEX_NUM = 5904
TRI_NUM = 11438
image_size = 96
chanel_color = 3
texture_size = [128,128]


# Training Parameters
learning_rate = 0.00005
num_steps = 10000 
display_step = 500
batch_size = 1

#------------------------------------------------------------------------
with tf.name_scope('input'):
    image_input = tf.placeholder(tf.float32, shape = [None, 224, 224, chanel_color],name = "image_input")
    shape_groundtruth = tf.placeholder(tf.float32, shape=[None, VERTEX_NUM,3], name = "shape_groundtruth")
    texture_groundtruth = tf.placeholder(tf.float32, shape=[None, texture_size[0], texture_size[1],3], name  = "texture_groundtruth")
    camera_groundtruth = tf.placeholder(tf.float32, shape=[None, m_dim], name = "camera_groundtruth")
    image_input_resized = tf.placeholder(tf.float32, shape = [None, 96, 96, chanel_color],name = "image_input_resized")
with tf.name_scope('output'): 
    #output of encoder
    cameraSetup_estimed = tf.placeholder(tf.float32, shape = [None, m_dim], name = "cameraSetup_estimed")
    shapeEncoding_estimed = tf.placeholder(tf.float32, shape = [None, dim_encoding_shape], name = "shapeEncoding_estimed")
    textureEncoding_estimed = tf.placeholder(tf.float32, shape = [None, dim_encoding_texture], name = "textureEncoding_estimed")
    #output of decoder
    shape_estimed = tf.placeholder(tf.float32, shape = [None,VERTEX_NUM,3], name = "shape_estimed")
    texture_estimed = tf.placeholder(tf.float32, shape = [None, texture_size[0], texture_size[1],3], name = "texture_estimed")

def autoencoder(image):
    with tf.variable_scope('autoencoder', reuse=tf.AUTO_REUSE):
        
        shapeEncoding, textureEncoding, cameraSetup = encoder(image)
        
        texture = decoder_texture(textureEncoding)
        
        shape = decoder_shape(shapeEncoding)
  
        return shape, texture , cameraSetup


#Construire le model
image_input_resized = tf.image.resize_images(image_input, [96,96])
shape_estimed, texture_estimed , cameraSetup_estimed = autoencoder(image_input_resized)
image_reconstruite,_ = warp_uvTexture(texture_estimed*255, cameraSetup_estimed, shape_estimed)   


#index of batch
index_batch = tf.constant(0)
#index of facteur of scale and rotation
index_rotation = tf.constant([0,1,2,3])
rotation_groundtruth = tf.gather(tf.gather(camera_groundtruth, index_batch), index_rotation)
rotation_estimed = tf.gather(tf.gather(cameraSetup_estimed, index_batch), index_rotation)
loss_rotation = tf.nn.l2_loss(tf.subtract(rotation_groundtruth , rotation_estimed))*1000


#index of landmark
data_landmark = np.load("./Mesh_definition/FaceGen_mesh_definition.npz")
data_landmark =  np.int32(data_landmark["landmark"])
index_landmark = tf.constant(data_landmark)
landmark_groundtruth = tf.gather(tf.gather(shape_groundtruth, index_batch), index_landmark)
landmark_estimed = tf.gather(tf.gather(shape_estimed, index_batch), index_landmark)
loss_landmark =  tf.nn.l2_loss(tf.subtract(landmark_estimed , landmark_groundtruth))/2268



loss_shape  = tf.nn.l2_loss(tf.subtract(shape_estimed , shape_groundtruth))/22680 #22680 is variance of a vector shape
loss_texture = tf.losses.absolute_difference(texture_estimed, texture_groundtruth)*255
loss_m = tf.nn.l2_loss(tf.subtract(cameraSetup_estimed , camera_groundtruth))

loss_pre_train = loss_shape + loss_texture + loss_m + loss_rotation + loss_landmark
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op_pre_train = optimizer.minimize(loss_pre_train)

saver = tf.train.Saver()


#------------------------------------------------------------------------------------------------------

def generator_batch(size_batch):
    indices = np.random.randint(199,size = size_batch)
    out_img = data_image[indices]
    out_shape = data_shape[indices]
    out_texture = data_texture[indices]
    out_m = data_camera[indices]
    return out_img, out_shape, out_texture, out_m



#------------------------------------------------------------------------------------------------------
with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
     #pre_training_model
    for step in range(num_steps):
        img, S, T, m = generator_batch(batch_size)
        feed_dict={image_input: np.array(img), shape_groundtruth: np.array(S), 
                                  texture_groundtruth: np.array(T/255), camera_groundtruth : np.array(m)}
        _, l, l_shape,l_texture, l_m = sess.run([train_op_pre_train, loss_pre_train,loss_shape,loss_texture,loss_m],feed_dict )
        
        if (step % display_step == 0 and step != 0) :
            image_input_i, image_estimted_i, texture_groundtruth_i, texture_estimed_i = sess.run([image_input,image_reconstruite,texture_groundtruth*255,texture_estimed*255],feed_dict )
            print('Step %i: Minibatch Loss: %f' % (step, l))
            print ('Step %i: Loss of shape: %f' % (step, l_shape))
            print ('Step %i: Loss of texture: %f' % (step, l_texture))
            print ('Step %i: Loss of camera: %f' % (step, l_m))
            print ('----------------------------------')
            if step >=2000:
                save_images(np.array([image_input_i[0],image_estimted_i[0]]), [1, 2], save_result_path+'image'+str(step)+'.png')
                save_images(np.array([texture_groundtruth_i[0],texture_estimed_i[0]]), [1, 2], save_result_path+'texture'+str(step)+'.png')

    saver.save(sess, save_pre_train_model_path)

