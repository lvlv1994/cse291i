#%%
import numpy as np
import scipy
import scipy.spatial
import scipy.optimize
import random
from sklearn.metrics.pairwise import euclidean_distances
#%%
#read teapot and violet data
tea_pot = open('pynycloud_teapot.txt','r')
violet = open('pynycloud_violet.txt','r')
def read_data(file1):
    s = []
    for line in file1:
        line = line.split(',')
        line = [float(i) for i in line[:-1]]    
        s.append(line)
    s =np.asarray(s) 
    
    file1.close()
    return s
    
teapot_data = read_data(tea_pot)
violet_data = read_data(violet)
#%%
#inbuilt scipy function
# compute D
D = euclidean_distances(teapot_data,violet_data)
row_ind, col_ind = scipy.optimize.linear_sum_assignment(D)
cost = np.sum(D[row_ind,col_ind])

#%%
#visualize row and col
import matplotlib.pyplot as plt
def visualize_graph(row,col,title):
    plt.scatter(row, col,c = 'b')
    plt.title(title)
    plt.show()
visualize_graph(row_ind,col_ind,'Distribution of points index calling Scipy')
    
#%%
from Q3_p1_prime import process
p = process(D)
[row_a,col_a] = p.step_running()
cost = np.sum(D[row_a,col_a])
#%%
visualize_graph(row_a,col_a,'Distribution of points index calling our implemented function')
    
 