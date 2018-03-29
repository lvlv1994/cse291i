#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 13:16:12 2018

@author: chunyilyu
"""

import tf_emddistance
import tensorflow as tf
import random
import numpy as np
num_s = 500
num_S = 100

#%%
# get S and s which is points of cloud points and sets of cloud point
def get_sets(num_s,num_S):
    random.seed(111)
    S = []
    for i in range(num_S):
        r = random.randint(1,10)
        s = []
        for j in range(num_s):
            angel = random.uniform(0,int(np.pi * 2))
            s.append([r*np.cos(angel),r*np.sin(angel),0])
        S.append(s)
    return S
#%%    
#build model    
S = tf.placeholder(tf.float32,[None,num_s,3])
x = tf.Variable(tf.truncated_normal([1,num_s,3],mean=0.0,stddev=0.1), dtype = tf.float32)
X = tf.tile(x,[tf.shape(S)[0],1,1])
dist,_,_ = tf_emddistance.emd_distance(S,X)
loss = tf.reduce_mean(dist)*10000
optimizer = tf.train.GradientDescentOptimizer(1e-3).minimize(loss)


#%%
#run model
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

num_iter = 100
batch_size = 64
loss = []
S_cloud = get_sets(num_s,num_S)
for i in range(num_iter):
    _,loss = sess.run([optimizer,loss],feed_dict = {S:S_cloud})
    if i % 10 == 0:
        print(loss)
results = sess.run(x,feed_dict = {S:S_cloud})
