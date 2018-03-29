"""
Helper functions to implement PointNet
"""
MODELNET40_PATH = "/Users/chunyilyu/pythoncode/cse291i/Homework3/modelnet40_ply_hdf5_2048/"
import numpy as np
import h5py
import os

def load_h5(h5_filename):
    """
    Data loader function.
    Input: The path of h5 filename
    Output: A tuple of (data,label)
    """
    f = h5py.File(h5_filename)
    print(f['data'])
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def get_category_names():
    """
    Function to list out all the categories in MODELNET40
    """
    shape_names_file = os.path.join(MODELNET40_PATH, 'shape_names.txt')
    shape_names = [line.rstrip() for line in open(shape_names_file)]
    return shape_names

def evaluate(true_labels,predicted_labels):
    """
    Function to calculate the total accuracy.
    Input: The ground truth labels and the predicted labels
    Output: The accuracy of the model
    """
    return np.mean(true_labels == predicted_labels)
