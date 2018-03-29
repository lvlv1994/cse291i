#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 16:24:55 2018

@author: chunyilyu
"""
#%%
import numpy as np
#import tensorflow as tf
import utils
#import pymesh
from pyntcloud import PyntCloud
import pandas as pd
#%% read data
train_cloud = []
train_label = []
dir_path = "/Users/chunyilyu/pythoncode/cse291i/Homework3/modelnet40_ply_hdf5_2048/"
category_names = utils.get_category_names()

def read_data(dir_path):
    train_cloud = []
    train_label = []
    for i in range(0,5):
        file_path = dir_path+ 'ply_data_train' + str(i) +'.h5'
        train_data0 = utils.load_h5(file_path)
        train_cloud.extend(np.array(train_data0[0]))
        train_label.extend(np.array(train_data0[1]))

    return np.array(train_cloud),np.array(train_label)

train_cloud,train_label = read_data(dir_path)

    




#%%

#cloud.to_file('vis_1.ply')

#%% rotation and jitter
'''def rotate(points,theta):
    rotation_matrix = np.array([[ np.cos(theta), 0, np.sin(theta)],
                                [ 0,             1,             0],
                                [-np.sin(theta), 0, np.cos(theta)]])
    rotated_pt_cloud = []
    for p in points:
         rotated_pt_cloud.append((np.matmul(rotation_matrix, p)))
    return np.array(rotated_pt_cloud)
def jitter(points,mean,std):
    return points+np.random.normal(mean, std, points.shape)

        
#%% define the model
points = tf.placeholder(tf.float32,[None,2048,3])
labels = tf.placeholder(tf.float32,[None,40])      #need to make things more generalize

layer_conv1 = tf.layers.conv1d(points,filters=64,kernel_size=1,strides=1)
layer_conv1 = tf.contrib.layers.batch_norm(layer_conv1)

layer_conv2 = tf.layers.conv1d(layer_conv1,filters=64,kernel_size=1,strides=1)
layer_conv2 = tf.contrib.layers.batch_norm(layer_conv2)

layer_conv3 = tf.layers.conv1d(layer_conv2,filters=64,kernel_size=1,strides=1)
layer_conv3 = tf.contrib.layers.batch_norm(layer_conv3)

layer_conv4 = tf.layers.conv1d(layer_conv3,filters=128,kernel_size=1,strides=1)
layer_conv4 = tf.contrib.layers.batch_norm(layer_conv4)

layer_conv5 = tf.layers.conv1d(layer_conv4,filters=1024,kernel_size=1,strides=1)
layer_conv5 = tf.contrib.layers.batch_norm(layer_conv5)

#%% maxpooling
layer_max = tf.reduce_max(layer_conv5,1)

#%%
layer_fnn1 = tf.contrib.layers.fully_connected(inputs=layer_max,num_outputs=512,activation_fn=tf.nn.relu)
layer_fnn2 = tf.contrib.layers.fully_connected(inputs=layer_fnn1,num_outputs=256,activation_fn=tf.nn.relu)
layer_fnn3 = tf.contrib.layers.fully_connected(inputs=layer_fnn2,num_outputs=40,activation_fn=tf.nn.relu)
output = tf.nn.softmax(layer_fnn3)
output_class = tf.argmax(output,axis=1)
#%% running the model
batch_size = 64
current_epoch = 0

if current_epoch != 0:
    saver = tf.train.import_meta_graph("models/PointNet_Vanilla-%i.meta" % current_epoch)
    saver.restore(sess, tf.train.latest_checkpoint("./models"))
    current_learning_rate /= np.power(2, current_epoch // 20)
else:
    saver = tf.train.Saver()

'''

