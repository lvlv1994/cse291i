import numpy as np
import glob
import tensorflow as tf
from os.path import basename,abspath

class Loader:
    def __init__(self, dataset="data/NYU.tfrecords", test_ratio=100, batch_size=8, allow_smaller_final_batch=False, seed=0):
    filename_queue = tf.train.string_input_producer([dataset], num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'depth': tf.FixedLenFeature([], tf.string),
            'image': tf.FixedLenFeature([], tf.string)
        })
    # now return the converted data
    label = tf.decode_raw(features['label'], tf.float32)
    image = tf.decode_raw(features['image'], tf.uint8)
    batch = tf.train.shuffle_batch([image, label], batch_size, allow_smaller_final_batch=allow_smaller_final_batch)
    
    def get_train_reader(self):
        reader11 = tf.WholeFileReader()
        reader12 = tf.WholeFileReader()
        _, train_imgs = reader11.read(self.__train_image_queue)
        _, train_depths = reader12.read(self.__train_depth_queue)
        train_imgs = tf.image.resize_image_with_crop_or_pad(
            tf.image.decode_image(train_imgs, channels=3), 480, 640)
        train_depths = tf.image.resize_image_with_crop_or_pad(
            tf.image.decode_image(train_depths, channels=1), 480, 640)
        train_imgs.set_shape([train_imgs.shape[0], train_imgs.shape[1], 3])
        train_depths.set_shape([train_depths.shape[0], train_depths.shape[1], 1])
        return tf.train.batch([train_imgs, train_depths], self.__bs, allow_smaller_final_batch=self.__smaller_final)
    
    def get_test_reader(self):
        reader11 = tf.WholeFileReader()
        reader12 = tf.WholeFileReader()
        _, test_imgs = reader11.read(self.__test_image_queue)
        _, test_depths = reader12.read(self.__test_depth_queue)
        test_imgs = tf.image.resize_image_with_crop_or_pad(
            tf.image.decode_image(test_imgs, channels=3), 480, 640)
        test_depths = tf.image.resize_image_with_crop_or_pad(
            tf.image.decode_image(test_depths, channels=1), 480, 640)
        test_imgs.set_shape([test_imgs.shape[0], test_imgs.shape[1], 3])
        test_depths.set_shape([test_depths.shape[0], test_depths.shape[1], 1])
        return tf.train.batch([test_imgs, test_depths], self.__bs, allow_smaller_final_batch=self.__smaller_final)
        
    def size(self):
        return len(self.__train_images), len(self.__test_images)
    
    def n_batches(self):
        if self.__smaller_final:
            op = np.ceil
        else:
            op = np.floor
        return int(op(len(self.__train_images)/self.__bs)), int(op(len(self.__test_images)/self.__bs))
    
    def batch_size(self):
        return self.__bs