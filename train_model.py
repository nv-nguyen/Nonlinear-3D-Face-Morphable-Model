#In this train model, we need to load the pre-train model and the data "CelebA_cropped" 
#We define also the path to save the result. By defaut, the data is in the same directory, the result is being saved in "./output/pre_train_model/"


from __future__ import division, print_function, absolute_import
import numpy as np
import tensorflow as tf
import os
import sys
from encoder import encoder 
from decoder import decoder_texture, decoder_shape 
from utils_in_out import save_images
from engine_rendering_FaceGen import warp_uvTexture
import matplotlib.pyplot as plt
from PIL import Image
import glob
#----------------------------------------------------------------------------------------
#save_path for image reconstruited
save_result_path = './output/train_model/'
#save_path for train_model
save_train_model_path = './save_model_train/my-test-model.ckpt'
#load pre_train model
pre_train_model_path = "./save_model_pre_train/my-test-model.ckpt" 
#load database CelebA
CelebA_PATH = '/home/van/Documents/data/croppedCelebA/van/*.jpg'

Filename_CelebA_PATH = glob.glob(CelebA_PATH)
#----------------------------------------------------------------------------------------
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
learning_rate =0.00005
num_steps = 19500*2
batch_size = 1
display_step = 1000

#----------------------------------------------------------------------------------------
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

loss_train = tf.losses.absolute_difference(image_reconstruite, image_input)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op_train = optimizer.minimize(loss_train)



saver = tf.train.Saver(max_to_keep = 2*20)

#---------------------------------------------------------------------------
def load_image_resize(addr):
    output = Image.open(addr)
    output = output.convert("RGB")
    output = output.resize((224,224), Image.ANTIALIAS)
    return np.array(output,np.float32)

def generator_batch(batch_size):
    indices = np.random.randint(int(len(Filename_CelebA_PATH)-1),size = batch_size)[0]
    return load_image_resize(Filename_CelebA_PATH[indices])
#-------------------------------------------------------------------------------------------------------------------------


with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    print ("Restoring the model pre-train...")
    saver.restore(sess, pre_train_model_path )
    print ("Now, the model is being trained...")
    for step in range(num_steps):
        # img = sess.run(images_train_tfrecords)
        img = generator_batch(batch_size)
        feed_dict={image_input: np.array([img])}
        _, l = sess.run([train_op_train, loss_train],feed_dict )
        if (step % display_step == 0 and step != 0) :
            image_correct, image_out, texture_out = sess.run([image_input,image_reconstruite, texture_estimed*255],feed_dict )
            print('Step %i: Minibatch Loss: %f' % (step, l))
            print ('----------------------------------')
            save_images(np.array([image_correct[0],image_out[0]]), [1, 2], save_result_path+'image'+str(step)+'.png')
            save_images(np.array([texture_out[0]]), [1, 1], save_result_path+'texture'+str(step)+'.png')
    if step % 10000 == 0 :
            saver.save(sess, save_train_model_path , global_step=step)  
    # Stop the threads
    sess.close()

