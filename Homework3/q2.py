#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 10:38:23 2018

@author: chunyilyu
"""

import numpy as np
import tensorflow as tf
import utils
#import pymesh
#from pyntcloud import PyntCloud
import pandas as pd
train_cloud = []
train_label = []
dir_path = "/datasets/home/55/755/cs291eau/PA3/modelnet40_ply_hdf5_2048/"
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

    

#%% rotation and jitter
def rotate(points):
    theta = np.random.uniform() * 2 * np.pi
    rotation_matrix = np.array([[ np.cos(theta), 0, np.sin(theta)],
                                [ 0,             1,             0],
                                [-np.sin(theta), 0, np.cos(theta)]])
    rotated_pt_cloud = []
    for p in points:
         rotated_pt_cloud.append((np.matmul(p,rotation_matrix)))
    return np.array(rotated_pt_cloud)
def jitter(points,mean=0,std=0.02):
    return points+np.random.normal(mean, std, points.shape)

def conv_layer(prev_layer, layer_depth, kernel_size, batch_norm, batch_norm_decay, scope, is_training=False):
    with tf.variable_scope(scope) as sc:
        strides = [1,1]
        conv_layer = tf.layers.conv2d(prev_layer, layer_depth, kernel_size, strides, use_bias=False, activation=None)
        if batch_norm:
            conv_layer = tf.layers.batch_normalization(conv_layer, momentum=batch_norm_decay, training=is_training)
        conv_layer = tf.nn.relu(conv_layer)
        return conv_layer
def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      0.5,
                      batch*32,
                      200000,
                      0.5,
                      staircase=True)
    bn_decay = tf.minimum(0.99, 1 - bn_momentum)
    return bn_decay

def input_transform_net(point_cloud, is_training, bn_decay=None, K=3):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return:
            Transformation matrix of size 3xK """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value

    pt_points = tf.expand_dims(point_cloud, -1)
    
    layer_conv1 = tf.layers.conv2d(inputs=pt_points, filters=64, kernel_size=[1,3], padding="VALID", 
                               activation=tf.nn.relu)
    layer_conv1 = tf.contrib.layers.batch_norm(layer_conv1,is_training = is_training,bn_decay=bn_decay)
    
    
    layer_conv2 = tf.layers.conv2d(inputs=pt_points, filters=64, kernel_size=[1,3], padding="VALID", 
                               activation=tf.nn.relu)
    layer_conv1 = tf.contrib.layers.batch_norm(layer_conv1,is_training = is_training,bn_decay=bn_decay)
 
    net = conv_layer(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
    net = conv_layer(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay)
    net = conv_layer(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv3', bn_decay=bn_decay)
    net = tf.nn.max_pool(net, [num_point,1],
                             padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf.contrib.layers.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    net = tf.contrib.layers.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_XYZ') as sc:
        assert(K==3)
        weights = tf.get_variable('weights', [256, 3*K],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [3*K],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        biases += tf.constant([1,0,0,0,1,0,0,0,1], dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, 3, K])
    return transform


def feature_transform_net(inputs, is_training, bn_decay=None, K=64):
    """ Feature Transform Net, input is BxNx1xK
        Return:
            Transformation matrix of size KxK """
    batch_size = inputs.get_shape()[0].value
    num_point = inputs.get_shape()[1].value

    net = conv_layer(inputs, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
    net = conv_layer(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay)
    net = conv_layer(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv3', bn_decay=bn_decay)
    net = tf.nn.max_pool(net, [num_point,1],
                             padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf.contrib.layers.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    net = tf.contrib.layers.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_feat') as sc:
        weights = tf.get_variable('weights', [256, K*K],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [K*K],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        biases += tf.constant(np.eye(K).flatten(), dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, K, K])
    return transform
def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      0.5,
                      batch*32,
                      200000,
                      0.5,
                      staircase=True)
    bn_decay = tf.minimum(0.99, 1 - bn_momentum)
    return bn_decay

def input_transform_net(point_cloud, is_training, bn_decay=None, K=3):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return:
            Transformation matrix of size 3xK """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value

    pt_points = tf.expand_dims(point_cloud, -1)
    
    layer_conv1 = tf.layers.conv2d(inputs=pt_points, filters=64, kernel_size=[1,3], padding="VALID", 
                               activation=tf.nn.relu)
    layer_conv1 = tf.contrib.layers.batch_norm(layer_conv1,is_training = is_training,bn_decay=bn_decay)
    
    
    layer_conv2 = tf.layers.conv2d(inputs=pt_points, filters=64, kernel_size=[1,3], padding="VALID", 
                               activation=tf.nn.relu)
    layer_conv1 = tf.contrib.layers.batch_norm(layer_conv1,is_training = is_training,bn_decay=bn_decay)
 
    net = conv_layer(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
    net = conv_layer(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay)
    net = conv_layer(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv3', bn_decay=bn_decay)
    net = tf.nn.max_pool(net, [num_point,1],
                             padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf.contrib.layers.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    net = tf.contrib.layers.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_XYZ') as sc:
        assert(K==3)
        weights = tf.get_variable('weights', [256, 3*K],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [3*K],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        biases += tf.constant([1,0,0,0,1,0,0,0,1], dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, 3, K])
    return transform


def feature_transform_net(inputs, is_training, bn_decay=None, K=64):
    """ Feature Transform Net, input is BxNx1xK
        Return:
            Transformation matrix of size KxK """
    batch_size = inputs.get_shape()[0].value
    num_point = inputs.get_shape()[1].value

    net = conv_layer(inputs, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
    net = conv_layer(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay)
    net = conv_layer(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv3', bn_decay=bn_decay)
    net = tf.nn.max_pool(net, [num_point,1],
                             padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf.contrib.layers.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    net = tf.contrib.layers.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_feat') as sc:
        weights = tf.get_variable('weights', [256, K*K],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [K*K],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        biases += tf.constant(np.eye(K).flatten(), dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, K, K])
    return transform
#%% define the model
learning_rate = 0.001
points = tf.placeholder(tf.float32,[None,2048,3])
#pt_points = tf.expand_dims(points, -1)
num_point = points.get_shape()[1].value
print(num_point)
batch = tf.Variable(0)
bn_decay = get_bn_decay(batch)
labels = tf.placeholder(tf.int32,[None])      #need to make things more generalize
is_training = tf.placeholder(tf.bool)

transform = input_transform_net(points, is_training, bn_decay, K=3)
X_new = tf.matmul(points, is_training, transform)
input_x = tf.expand_dims(X_new, -1)


layer_conv1 = tf.layers.conv2d(inputs=input_x, filters=64, kernel_size=[1,3], padding="VALID", activation=tf.nn.relu)
layer_conv1 = tf.contrib.layers.batch_norm(layer_conv1)

layer_conv2 = tf.layers.conv2d(inputs=layer_conv1, filters=64, kernel_size=[1,1], padding="VALID", 
                                       activation=tf.nn.relu)
layer_conv2 = tf.contrib.layers.batch_norm(layer_conv2)

with tf.variable_scope('transform_net2') as sc:
    transform = feature_transform_net(conv_layer2, is_training, bn_decay, K=64)

layer_conv3 =  tf.layers.conv2d(inputs=layer_conv2, filters=64, kernel_size=[1, 1], padding="VALID", 
                                       activation=tf.nn.relu)
layer_conv3 = tf.contrib.layers.batch_norm(layer_conv3)

layer_conv4 = tf.layers.conv2d(inputs=layer_conv3, filters=128, kernel_size=[1, 1], padding="VALID", 
                                       activation=tf.nn.relu)
layer_conv4 = tf.contrib.layers.batch_norm(layer_conv4)

layer_conv5 = tf.layers.conv2d(inputs=layer_conv4, filters=1024, kernel_size=[1, 1], padding="VALID", 
                                       activation=tf.nn.relu)
layer_conv5 = tf.contrib.layers.batch_norm(layer_conv5)


#%% maxpooling
layer_max = tf.nn.max_pool(layer_conv5, ksize=[1,num_point,1,1], strides=[1,2,2,1], padding='VALID')
layer_global = tf.reshape(layer_max,[-1,1024])
#%%
layer_fnn1 = tf.contrib.layers.fully_connected(inputs=layer_global,num_outputs=512,activation_fn=tf.nn.relu)
layer_fnn2 = tf.contrib.layers.fully_connected(inputs=layer_fnn1,num_outputs=256,activation_fn=tf.nn.relu)
layer_fnn3 = tf.contrib.layers.fully_connected(inputs=layer_fnn2,num_outputs=40,activation_fn=tf.nn.relu)

#output = tf.nn.softmax(layer_fnn3)
#output_class = tf.argmax(output,axis=1)
print(np.shape(layer_fnn3),np.shape(labels))

#define loss function
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=layer_fnn3, labels=labels)
loss = tf.reduce_mean(loss)

#define prediction
predictions = tf.argmax(tf.nn.softmax(layer_fnn3), axis=1)

#define accuarcy
predict = tf.cast(tf.argmax(layer_fnn3,1),tf.int32)
correct_prediction = tf.equal(predict, labels)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#define optimazer
optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

#%% running the model
# Create session
import os
batch_size = 32
current_epoch = 0
num_iter = 100

LOG_DIR = 'log_'
list_loss = []
list_acc = []
num_train = train_label.shape[0]
num_test = test_label.shape[0]
#learning_rate = 0.001
#rate = tf.Variable(0.001)
rate = 0.001

with tf.Session() as sess:
    if current_epoch != 0:
        saver = tf.train.import_meta_graph("models/PointNet_beta-%i.meta" % current_epoch)
        saver.restore(sess, tf.train.latest_checkpoint("./models"))
    else:
        saver = tf.train.Saver()
    #merged = tf.summary.merge_all()
    #train_writer =  tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),sess.graph)
    sess.run(tf.global_variables_initializer())

    for epoch in range(num_iter):
        if epoch %20 == 0:
            rate /= 2
            #rate = tf.Variable(learning_rate)
        loss_train_all = 0
        acc_train_all = 0
        idx = np.arange(num_train)
        np.random.shuffle(idx)
        train_cloud = train_cloud[idx, ...]
        train_label = train_label[idx]
        batch_num = np.ceil(num_train/batch_size).astype(int)
        batch_num_test = np.ceil(num_test/batch_size).astype(int)
        for batch_idx in range(batch_num):
            start_idx = batch_idx*batch_size
            end_idx = np.min([(batch_idx+1)*batch_size,num_train-1])
            batch_img = train_cloud[start_idx:end_idx,...]
            
            #print(np.shape(batch_img))
            batch_y = train_label[start_idx:end_idx,...]
            #batch_y = train_label_prev[start_idx:end_idx,...]
            #print(np.shape(batch_y))
            batch_y = np.reshape(batch_y,[-1])
            rotation_data = rotate(batch_img)
            augment_data = jitter(rotation_data)
            #print(np.shape(augment_data))
            sess.run([model_train],feed_dict={points: augment_data, labels: batch_y, is_training: True,\
                                              learning_rate:rate,drop_rate : 0.7})            
            loss_train,acc_train = sess.run([loss,accuracy], \
                                        feed_dict={points: augment_data, labels: batch_y,is_training: False,\
                                                  learning_rate:rate,drop_rate : 0.7},)
            #print(loss_train,acc_train)
            loss_train_all += loss_train*(end_idx-start_idx)
            acc_train_all += acc_train * (end_idx-start_idx)
            #print(loss_train_all,acc_train_all)
        
        
        loss_train_all = loss_train_all / num_train
        acc_train_all = acc_train_all / num_train  
        list_loss.append(loss_train_all)
        list_acc.append(acc_train_all)
        print('TRAIN: epoch: ', epoch+1, '\tloss: %.4f'%loss_train_all, '\taccuracy: %.4f'%acc_train_all)
        if (epoch+1) % 20 == 0:
            save_path = saver.save(sess,"models/PointNet_Vanilla", global_step=current_epoch)
            print('TRAIN: epoch: ', epoch+1, '\tloss: %.4f'%loss_train_all, '\taccuracy: %.4f'%acc_train_all)
        loss_test_all = 0
        acc_test_all = 0
        '''for batch_idx in range(batch_num_test):
            start_idx = batch_idx*batch_size
            end_idx = np.min([(batch_idx+1)*batch_size,num_train-1])
            batch_img = test_cloud[start_idx:end_idx,...]
            batch_y = test_label[start_idx:end_idx]
            batch_y = np.reshape(batch_y,[-1])
            rotation_data = rotate(batch_img)
           
            #sess.run([optim],{points: augment_data, labels: batch_y,is_training: True}) 
           
            loss_test,acc_test = sess.run([loss,accuracy], \
                                        feed_dict={points: rotation_data, labels: batch_y,is_training: False})
            loss_test_all = loss_test_all + loss_test*(end_idx-start_idx)
            acc_test_all = acc_test_all + acc_test*(end_idx-start_idx) 
        
        loss_test_all = loss_test_all / num_test
        acc_test_all = acc_test_all / num_test        
        print('Test: epoch: ', epoch+1, '\tloss: %.4f'%loss_test_all, '\taccuracy: %.4f'%acc_test_all)
        if (epoch+1) % 50 == 0:
            #save_path = saver.save(sess,"models/PointNet_Vanilla", global_step=current_epoch)
            print('TEST: epoch: ', epoch+1, '\tloss: %.4f'%loss_test_all, '\taccuracy: %.4f'%loss_test_all)

           
        
        #idx = np.random.choice(2048,[batch_size],False)
        #batch_img = train_cloud[idx][:]
        #batch_y = train_label[idx]
        #batch_y = np.reshape(batch_y,[-1])
        
        
        #print(np.shape(rotation_data))
        
        #print(np.shape(augment_data),np.shape(batch_y))
       
        #train_writer.add_summary(summary, log_step)
        if (epoch+1) % 50 == 0:
            save_path = saver.save(sess,"models/PointNet_Vanilla", global_step=current_epoch)
            print('TRAIN: epoch: ', epoch+1, '\tloss: %.4f'%loss_train, '\taccuracy: %.4f'%acc_train)
        
        '''
    

