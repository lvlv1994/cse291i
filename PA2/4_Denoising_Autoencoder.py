#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 23:41:14 2018

@author: chunyilyu
"""

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import random
#%%
#Get data
mnist = input_data.read_data_sets(".",one_hot = True)
#%%
#Add gaussian noise
images = mnist.train.images + np.random.normal(0,0.05,(55000,784))
#%%
#build encode model
image = tf.placeholder(tf.float32,[None,784])
image_reshape = tf.reshape(image,[-1,28,28,1])
noisy = tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=0.2, dtype=tf.float32)
print(noisy.get_shape())
image_noisy = noisy + image
print(image_noisy.get_shape())
X = tf.reshape(image_noisy,[-1,28,28,1])
print(X.get_shape())
E1 = tf.layers.conv2d(X, filters = 8, kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu)

print(E1.get_shape())
#12*12*8   
E2 = tf.layers.conv2d(E1, filters = 4, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)

print(E2.get_shape())
#5*5*4
#%%
#build decode model
E2 = tf.reshape(E2, [-1, 10, 10, 1])
print(E2.get_shape())
D1 = tf.layers.conv2d_transpose(E2,filters=64,kernel_size=5,strides=1,padding="valid",activation=tf.nn.relu)
print(D1.get_shape())
#14*14*64
D2 = tf.layers.conv2d_transpose(D1,filters=1,kernel_size=15,strides=1,padding="valid",activation=tf.nn.relu)
print(D2.get_shape())
#28*28*1
#%%
#get loss
loss = tf.reduce_mean(tf.subtract(image_reshape,D2)**2)
#print(tf.subtract(image_reshape,D2))
#%%
#get optimazer
optim = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
#%%
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


num_iter = 500
batch_size = 64
loss_set = []
for i in range(num_iter):
    batch_x,_ = mnist.train.next_batch(batch_size)

    image_reshape2,_, l, result_test = sess.run([image_reshape,optim, loss, D2], feed_dict={image: batch_x})
    loss_set.append(l)
    #print (l)
#%%
import matplotlib.pyplot as plt

def plot_graph(loss):
    y = [i for i in range(len(loss))]
    plt.plot(y,loss)
    plt.show()
plot_graph(loss_set)
#%%
#test data
test_ind = [random.sample(np.where(mnist.test.labels==1)[0],1)[0] for i in range(10)]
test_img_original = mnist.test.images[test_ind]
test_img_noisy = test_img_original + np.random.normal(scale = 0.2, size = (test_img_original.shape))
test_img_reconstructed = sess.run(D2,feed_dict={image:test_img_noisy})

#%%
def vis_number(ori,noi,recons):
    fig, axes = plt.subplots(nrows=3, ncols=10, sharex=True, sharey=True, figsize=(20,4))
    for images,row in zip([ori,noi,recons],axes):
        for img,ax in zip(images,row):
            ax.imshow(img.reshape((28,28)))
    fig.tight_layout(pad=0.1)
    plt.savefig('Q4.png')
vis_number(test_img_original,test_img_noisy,test_img_reconstructed)
    
    
    
    