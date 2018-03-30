import h5py
import numpy as np

class Loader:
    def __init__(self, file='./nyu_depth_v2_labeled.mat', train_ratio=0.9, batch_size=8, allow_smaller_final_batch=False):
        self.__datafile = h5py.File(file)
        self.__indices = np.arange(self.__datafile['images'].shape[0])
        np.random.shuffle(self.__indices)
        cut_idx = int(train_ratio*self.__indices.shape[0])
        self.__train_indices = self.__indices[:cut_idx]
        self.__test_indices = self.__indices[cut_idx:]
        self.__bs = batch_size
        self.__train_pool = np.array(self.__train_indices)
        self.__test_pool = np.array(self.__test_indices)
        self.__smaller_final = allow_smaller_final_batch
    
    @staticmethod
    def __get_batch(indices, batch_size):
        return np.random.choice(indices, size=batch_size, replace=False)
    @staticmethod
    def __read_batch(batch, f):
        return f['images'][np.sort(batch),...].transpose((0,3,2,1)), f['depths'][np.sort(batch),...].transpose((0,2,1))
    
    def train_batch(self):
        bs = self.__bs
        if self.__train_pool.shape[0]<self.__bs:
            if self.__smaller_final:
                bs = self.__train_pool.shape[0]
            else:
                self.__train_pool = np.array(self.__train_indices)
        batch = self.__get_batch(self.__train_pool, bs)
        self.__train_pool = np.setdiff1d(self.__train_pool, batch, assume_unique=True)
        if self.__train_pool.shape[0]==0:
            self.__train_pool = np.array(self.__train_indices)
        return self.__read_batch(batch, self.__datafile)
    
    def test_batch(self):
        bs = self.__bs
        if self.__test_pool.shape[0]<self.__bs:
            if self.__smaller_final:
                bs = self.__test_pool.shape[0]
            else:
                self.__test_pool = np.array(self.__test_indices)
        batch = self.__get_batch(self.__test_pool, bs)
        self.__test_pool = np.setdiff1d(self.__test_pool, batch, assume_unique=True)
        if self.__test_pool.shape[0]==0:
            self.__test_pool = np.array(self.__test_indices)
        return self.__read_batch(batch, self.__datafile)
    
    def size(self):
        return len(self.__train_indices), len(self.__test_indices)
    
    def n_batches(self):
        if self.__smaller_final:
            op = np.ceil
        else:
            op = np.floor
        return int(op(len(self.__train_indices)/self.__bs)), int(op(len(self.__test_indices)/self.__bs))