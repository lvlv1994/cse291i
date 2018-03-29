#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 13:37:06 2018

@author: chunyilyu
"""
import numpy as np
file1 = open('pynycloud_teapot.txt','r')
s = []
for line in file1:
    line = line.split(',')
    print(line)
    line = [float(i) for i in line[:-1]]
    
    #new_line = [float(i) for i in new_line]
    
    s.append(line)
    
file1.close()
    
s =np.asarray(s)   
#%%

from pyntcloud import PyntCloud
import pandas as pd
points = pd.DataFrame(s, columns=['x', 'y', 'z'])
#points[['red', 'blue', 'green']] = pd.DataFrame(colors, index=points.index)
cloud = PyntCloud(points)
cloud.plot(point_size = 0.01,lines=[], line_color=[])

#%%
file2 = open('Q3_circle.txt','r')
S_cloud = []
for line in file2:
    line = line.split(',')
    print(line)
    line = [float(i) for i in line[:-1]]
    
    #new_line = [float(i) for i in new_line]
    
    s.append(line)
    
file2.close()
    
S_cloud =np.asarray(s) 
#%%
import pandas as pd
from pyntcloud import PyntCloud
import pandas as pd
red_color = np.tile(np.array([255,0,0]),[50,1])
grey_color = np.tile(np.array([255,255,255]),[100*50,1])
colors = (np.vstack([red_color,grey_color])).astype(np.uint8)
P = np.vstack([s,np.reshape(S_cloud,[100*50,3])])
points = pd.DataFrame(P, columns=['x', 'y', 'z'])
points[['red', 'blue', 'green']] = pd.DataFrame(colors, index=points.index)
cloud = PyntCloud(points)