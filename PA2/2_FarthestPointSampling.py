#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 00:51:59 2018

@author: chunyilyu
"""

import pymesh
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
#%%
mesh = pymesh.load_mesh('teapot.obj')
#%%
#Get area
def triangle_area(mesh,face):
    f0,f1,f2 = face
    v0 = mesh.vertices[f0]
    v1 = mesh.vertices[f1]
    v2 = mesh.vertices[f2]
    return np.linalg.norm(np.cross(np.array(v1) - np.array(v0),np.array(v2) - np.array(v0)))/2
#%%
#Get weight
def get_weight(mesh):
    triangle_weight = []
    triangle_node = []
    for face in mesh.faces:
        triangle_node.append([mesh.vertices[face[0]],mesh.vertices[face[1]],mesh.vertices[face[2]]])
        triangle_weight.append(triangle_area(mesh,face))
        
    triangle_weight = np.asanyarray(triangle_weight)/sum(triangle_weight)
    return triangle_weight,triangle_node
triangle_weight,triangle_node = get_weight(mesh)
#%%
#get points cloud
total_numbers = 20000
P = []
for nodes, weight in zip(triangle_node,triangle_weight):
    num_points = weight * total_numbers
    for i in range(int(num_points)):
        r1 = np.random.rand()
        r2 = np.random.rand()
        D = (1-np.sqrt(r1))*nodes[0] + np.sqrt(r1)*(1-r2)*nodes[1]+np.sqrt(r1)*r2*nodes[2]
        P.append(D)
P = np.asarray(P)
#%%
#compute distance matrix
euclidean_distance = euclidean_distances(P,P)
euclidean_distance = np.asarray(euclidean_distance)
#%%
#create a point set S
import random
S_ind = random.sample(range(len(euclidean_distance)),1)
#%%
#compute on-mesh distance
#import sys
k = 500
S_index = random.sample(range(len(P)),1)
for iters in range(k-1):
    min_distance = np.min(euclidean_distance[S_index,],0)
    index_max = np.argmax(min_distance)
    S_index.extend([index_max])
S = P[S_index]
#%%
filep = open('pynycloud_teapot.txt','w')
for i in S:
    strs = ''
    for s in i:
        strs+= str(s) + ','
    filep.write(strs) 
    filep.write('\n')
filep.close()
#%%
#visulize
'''
new_tri = pymesh.tetgen()
new_tri.points = S
new_tri.run()
new_tea = new_tri.mesh
pymesh.save_mesh('new_teapot.obj',new_tea,use_float = True)
'''
from pyntcloud import PyntCloud
import pandas as pd
points = pd.DataFrame(S, columns=['x', 'y', 'z'])
cloud = PyntCloud(points)
cloud.plot(lines=[], line_color=[])
    
            
    
        
        
    


