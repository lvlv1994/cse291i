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
num_s = 50
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
            angel = random.uniform(0,np.pi * 2)
            s.append([r*np.cos(angel),r*np.sin(angel),0])
        S.append(s)
    return S
#%%    
#build model    
S = tf.placeholder(tf.float32,[None,num_s,3])
x = tf.Variable(tf.truncated_normal([1,num_s,3],mean=0.0,stddev=0.1), dtype = tf.float32)
X = tf.tile(x,[tf.shape(S)[0],1,1])
dist,_,_ = tf_emddistance.emd_distance(S,X)
print(dist)
loss = tf.reduce_mean(dist)*10000
train = tf.train.GradientDescentOptimizer(1e-3).minimize(loss)


#%%
#run model
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

num_iter = 200
batch_size = 64
loss = []
S_cloud = get_sets(num_s,num_S)
for i in range(num_iter):
    _,loss = sess.run([train,loss],feed_dict = {S:S_cloud})
    print(i,loss)
results = sess.run(x,feed_dict = {S:S_cloud})
results = np.reshape(results,[num_s,3])
print(results)
'''
filep = open('/Users/chunyilyu/pythoncode/cse291i/PA2/Q3.txt','w')
for i in results:
    strs = ''
    for s in i:
        strs+= str(s) + ','
    filep.write(strs)
    filep.write('\n')
filep.close()'''
'''
filep = open('/Users/chunyilyu/pythoncode/cse291i/PA2/Q3_circle.txt','w')
for i in S_cloud:
    strs = ''
    for s in i:
        for ss in s:
            strs+= str(ss) + ','
    filep.write(strs)
    filep.write('\n')
filep.close()
'''
import matplotlib.pyplot as plt
x = []
y = []
for r in results:
    x.append(r[0])
    y.append(r[1])
plt.scatter(x, y,c = 'r')
for s in S_cloud:
    x_1 = []
    y_1 = []
    for ss in s:
        x_1.append(ss[0])
        y_1.append(ss[1])
    plt.scatter(x_1, y_1,c = 'b')




plt.show()
