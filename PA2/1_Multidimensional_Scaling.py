#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 14:00:21 2018

@author: chunyilyu
"""
#%%
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
#from sklearn import manifold
import tensorflow as tf
import matplotlib.pyplot as plt
import random
N = 1000
#%%
#Read Data
mnist = input_data.read_data_sets("MNIST_data/")
#%%
samples = [[] for i in range(10)]
random.seed(11)
for image, label in zip(mnist.train.images, mnist.train.labels):
    if len(samples[label]) < N:
        samples[label].append(image)

#subproblem1 finished
#flatten and get labels
samples = [i for s in samples for i in s ]
labels = [i for i in range(10) for _ in range(1000)]
samples = np.asarray(samples)
#%%
#subproblen2 finished
#get D_ij
euclidean_distance = euclidean_distances(samples,samples)

#%%
#define the model
#init D
tf.reset_default_graph() 
D = tf.placeholder(tf.float32,[10*N,10*N])

#%%
#init X_2D\

X_2D = tf.get_variable("X", initializer=tf.random_normal((10*N, 2), stddev=0.02))
#%%
# generate x_i-x_j matrix referenced by Source: https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
L2 = tf.reduce_sum(tf.square(X_2D),1)
L2 = tf.reshape(L2,[-1,1])
D_ij = L2 - 2 * tf.matmul(X_2D,tf.transpose(X_2D)) + tf.transpose(L2)

#Define loss function
loss = tf.reduce_sum(tf.square(tf.sqrt(D_ij + 0.01) - D)) / 2 / 10000
#optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
#%% start cnn
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
num_iter = 1000
loss_values = []

for i in range(num_iter):
    _,l,X = sess.run([optimizer,loss,X_2D],feed_dict = {D:euclidean_distance})
    print(l)
    loss_values.append(l)
#%%
x_value = sess.run(X_2D)

#%%
#define ploting graph
def plot_graph(data,label):
    fig = plt.figure(figsize=[8,8])
    ax = fig.add_subplot(1,1,1)
    colors=['red','blue','green','yellow','gray','pink','olive','purple','cyan','orange']
    for i in range(len(data)):
        ax.text(data[i,0],data[i,1],str(label[i]),color = colors[label[i]])
    x_range = np.max(np.abs(label))
    plt.xlim((-10, 10))
    plt.ylim((-10, 10))
    plt.savefig('Q1_final.png')
    plt.show()
#plot_graph(x_value,labels)
plot_graph(x_value, labels)    
    
        
        
        
    



