#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 14:37:27 2018

@author: chunyilyu
"""
import numpy as np
class process():
    def __init__(self,cost_matrix):
        self.P = HungarianState(cost_matrix)
    def step_running(self):
        step = self.P.step1()
        row_ind = np.where(self.P.marked == 1)[0]
        col_ind = np.where(self.P.marked == 1)[1]
        return row_ind,col_ind
class HungarianState():
    def __init__(self,cost_matrix):
        self.cost_matrix = cost_matrix.copy()
        n,m = self.cost_matrix.shape
        self.uncovered_row = np.ones(n,dtype=bool)  #marked whether row is covers
        self.uncovered_col = np.ones(m,dtype=bool)  #marked whether row is covers
        self.marked = np.zeros((n,m),dtype=int)     #makred each cells
        self.row = 0
        self.col = 0
        self.path = np.zeros((n + m, 2), dtype=int)
    def making_zeros(self):
        self.uncovered_row[:] = True
        self.uncovered_col[:] = True
    def step1(self):
       #subtract smallest element in each row
       self.cost_matrix -= self.cost_matrix.min(axis=1)[:np.newaxis]
       
       #iterate each cells, if there is 0 in this cell, covered it as starred
       for i,j in zip(*np.where(self.cost_matrix == 0)):
           if self.uncovered_row[i] and self.uncovered_col[j]:
               self.marked[i,j] == 1
               self.uncovered_row[i] = False
               self.uncovered_row[i] = False
      
       self.making_zeros()
       self.step2()
    def step2(self):
        #Check each column, if every column here is 0, we are done, otherwise, go to step3
        marked = (self.marked==1)
        self.uncovered_col[np.any(marked, axis=0)] = False
        if marked.sum() < self.cost_matrix.shape[0]:
            #print('here')
            self.step3()
       
        return True
    def step3(self):
        #find uncorvered zero, if can't find any, go to next step; otherwise, cover this row and uncover the col containint zero.
        #continue this until no uncovered zeros left
        
        C = (self.cost_matrix == 0).astype(np.int)
        covered_C = C * self.uncovered_row[:, np.newaxis]
        covered_C *= self.uncovered_col.astype(dtype=np.int, copy=False)
        n,m = self.cost_matrix.shape[0],self.cost_matrix.shape[1]
        while True:
            row, col = np.unravel_index(np.argmax(covered_C), (n, m))
            if covered_C[row, col] == 0:
                    if np.any(self.uncovered_row) and np.any(self.uncovered_col):
                        minval = np.min(self.cost_matrix[self.uncovered_row], axis=0)
                        minval = np.min(minval[self.uncovered_col])
                        self.cost_matrix[np.logical_not(self.uncovered_row)] += minval
                        self.cost_matrix[:, self.uncovered_col] -= minval
                        self.step3()
            else:
                self.marked[row, col] = 2
                star_col = np.argmax(self.marked[row] == 1)
                if not self.marked[row, star_col] == 1:
                    # Could not find one
                    self.row = row
                    self.col = col
                    self.step4()
                else:
                    col = star_col
                    self.uncovered_row[row] = False
                    self.uncovered_col[col] = True
                    covered_C[:, col] = C[:, col] * (
                        self.uncovered_row.astype(dtype=np.int, copy=False))
                    covered_C[row] = 0
    def step4(self):
        
        count = 0
        path = self.path
        path[count, 0] = self.row
        path[count, 1] = self.col
    
        while True:
            # Find the first starred element in the col defined bythe path.
            row = np.argmax(self.marked[:, path[count, 1]] == 1)
            if not self.marked[row, path[count, 1]] == 1:
                break
            else:
                count += 1
                path[count, 0] = row
                path[count, 1] = path[count - 1, 1]
    
            # Find the first prime element in the row defined by the
            # first path step
            col = np.argmax(self.marked[path[count, 0]] == 2)
            if self.marked[row, col] != 2:
                col = -1
            count += 1
            path[count, 0] = path[count - 1, 0]
            path[count, 1] = col
    
        # Convert paths
        for i in range(count + 1):
            if self.marked[path[i, 0], path[i, 1]] == 1:
                self.marked[path[i, 0], path[i, 1]] = 0
            else:
                self.marked[path[i, 0], path[i, 1]] = 1
    
        self.making_zeros()
        # Erase all prime markings
        self.marked[self.marked == 2] = 0
        self.step2()


