#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 16:48:26 2018

@author: chunyilyu
"""
from numpy import linalg as LA
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
#%%
#adjancent matrix
chain = nx.grid_graph([30])
A = nx.to_numpy_array(chain)
#%%
#degree matrix
D = np.diag(np.sum(A,axis = 1))
#%%
L = D - A

#%%
#eigenvector
w,v = np.linalg.eigh(L)
v = v.T
#%%
#sort 
results = np.argsort(w)
#%%
fig, axes = plt.subplots(nrows=5, ncols=6, sharex=True, sharey=True, figsize=(20,4))
i = 0
for v_i,ax in zip(v,axes):
    for col in ax: 
    
        pos = nx.spectral_layout(chain)
        nx.draw_networkx(chain, pos=pos, node_color=v_i, font_size=0)
        col.set_axis_off()
        
    

fig.tight_layout(pad=0.1)
plt.savefig('Q1.png')
#%%
pos = nx.spectral_layout(chain)
nx.draw_networkx(chain, pos=pos, node_color=v[results[0]], font_size=0)

#%%
lattice = nx.grid_graph([30,30])

#%%
#Adjance matrix
A = nx.to_numpy_array(lattice)
#%%
#degree matrix

D = np.diag(np.sum(A,axis = 1))

#%%
L = D - A
w,v = np.linalg.eigh(L)
v = v.T
#%%
pos = nx.spectral_layout(lattice)
nx.draw_networkx(lattice, pos=pos, node_color=v[0], font_size=0)

#%%
fig, axes = plt.subplots(nrows=5, ncols=6, sharex=True, sharey=True, figsize=(20,4))
for v_i,ax in zip(v,axes):
    nx.draw_networkx(lattice, pos=pos, node_color=v[0], font_size=0)
    ax.set_axis_off()
    #ax.plot(recon)
    #ax.imshow(image)
fig.tight_layout(pad=0.1)
plt.savefig('Q2.png')
#%%
from PIL import Image
path = './dog.jpeg'
img_raw = Image.open(path)
img = np.array(img_raw.resize((150, 150), Image.ANTIALIAS))
gray_scale = np.dot(img[...,:3], [0.299, 0.587, 0.114])
plt.imshow(gray_scale,cmap='gray')

#%%
lattice = nx.grid_graph([150,150])
L = nx.laplacian_matrix(lattice)
#%%
la = L.toarray()
#%%

w,v = np.linalg.eigh(la)
v = v.T
#%%
import seaborn as sns
ax = sns.heatmap(np.reshape(v[0], [150, 150]), cmap="Greens")
#%%
E = v[-30:]
proj = np.dot(np.reshape(gray_scale, [1,22500]), E.T)
recon = np.dot(proj, E)

#%%
plt.imshow(np.reshape(recon, [150, 150]),cmap='gray')

