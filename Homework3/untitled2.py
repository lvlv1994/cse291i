#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 13:39:19 2018

@author: chunyilyu
"""
import tensorflow as tf
import utils
import numpy as np
# load training data and labels
data0 = utils.load_h5("/Users/chunyilyu/pythoncode/cse291i/Homework3/modelnet40_ply_hdf5_2048/ply_data_train0.h5")
data1 = utils.load_h5("/Users/chunyilyu/pythoncode/cse291i/Homework3/modelnet40_ply_hdf5_2048/ply_data_train1.h5")
data2 = utils.load_h5("/Users/chunyilyu/pythoncode/cse291i/Homework3/modelnet40_ply_hdf5_2048/ply_data_train2.h5")
data3 = utils.load_h5("/Users/chunyilyu/pythoncode/cse291i/Homework3/modelnet40_ply_hdf5_2048/ply_data_train3.h5")
data4 = utils.load_h5("/Users/chunyilyu/pythoncode/cse291i/Homework3/modelnet40_ply_hdf5_2048/ply_data_train4.h5")

train_data = data0[0]
print(np.shape(train_data))

train_labels = data0[1]
catagory_names = utils.get_category_names()
print(np.shape(train_labels))
# load test data
test0 = utils.load_h5("ply_data_test0.h5")
test1 = utils.load_h5("ply_data_test1.h5")
# aggregate test data, test label
test_data = np.append(test0[0], test1[0], axis=0)
test_labels = np.append(test0[1], test1[1], axis=0)

# one hot encode train_labels
train_labels = []
for l in data0[1]:
    one_hot = np.zeros(40, dtype=np.int)
    one_hot[l[0]] = 1
    train_labels.append(one_hot)
train_labels = np.array(train_labels)

test_labels_one_hot = []
for l in test_labels:
    one_hot = np.zeros(40, dtype=np.int)
    one_hot[l[0]] = 1
    test_labels_one_hot.append(one_hot)
test_labels_one_hot = np.array(test_labels_one_hot)

cloud = tf.placeholder(tf.float32, [None, 2048, 3])
pt_cloud = tf.expand_dims(cloud, -1)
print(np.shape(pt_cloud))

# placeholder for one-hot labels
y = tf.placeholder(tf.float32, [None, 40])

# placeholder for labels
y_labels = tf.placeholder(tf.int64, [None])

# 1st mlp layer
layer_conv1 = tf.contrib.layers.conv2d(inputs=pt_cloud, num_outputs=64, kernel_size=[1, 3], padding="VALID", activation_fn=tf.nn.relu)
print(np.shape(layer_conv1))

# 2nd mlp layer
layer_conv2 = tf.contrib.layers.conv2d(inputs=layer_conv1, num_outputs=64, kernel_size=[1, 1], activation_fn=tf.nn.relu)
print(np.shape(layer_conv2))

# 3rd mlp layer
layer_conv3 = tf.contrib.layers.conv2d(inputs=layer_conv2, num_outputs=64, kernel_size=[1, 1], activation_fn=tf.nn.relu)
print(np.shape(layer_conv3))

# 4th cnn
layer_conv4 = tf.contrib.layers.conv2d(inputs=layer_conv3, num_outputs=128, kernel_size=[1, 1], activation_fn=tf.nn.relu)
print(np.shape(layer_conv4))

# 5th cnn
layer_conv5 = tf.contrib.layers.conv2d(inputs=layer_conv4, num_outputs=1024, kernel_size=[1, 1], activation_fn=tf.nn.relu)
print(np.shape(layer_conv5))
# max pooling
max_pool = tf.contrib.layers.max_pool2d(inputs=layer_conv5, kernel_size=[2048, 1])

print(np.shape(max_pool))

# fnn1
layer_fnn1 = tf.contrib.layers.fully_connected(inputs=max_pool, num_outputs=512, activation_fn=tf.nn.relu)
print(np.shape(layer_fnn1))

# fnn2
layer_fnn2 = tf.contrib.layers.fully_connected(inputs=layer_fnn1, num_outputs=256, activation_fn=tf.nn.relu)
print(np.shape(layer_fnn2))

# fnn3
logits = tf.contrib.layers.fully_connected(inputs=layer_fnn2, num_outputs=40, activation_fn=tf.nn.relu)
logits = tf.squeeze(logits, [1, 2])
print(np.shape(logits))

# softmax
output = tf.nn.softmax(logits)
output_class = tf.argmax(output,axis=1)
print(np.shape(output))
print(np.shape(output_class))
# loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y))

# optimizer
optim = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
num_iter = 1000
batch_size = 32

for i in range(num_iter):
    idx = np.random.choice(2048, [batch_size], False)
    batch_img = train_data[idx][:]
    batch_y = train_labels[idx][:]
#     batch_img, batch_y = data.train.next_batch(batch_size)
    _, l= sess.run([optim, loss], feed_dict = {cloud: batch_img , y: batch_y})
    if i % 10 == 0:
        print(l)


                    