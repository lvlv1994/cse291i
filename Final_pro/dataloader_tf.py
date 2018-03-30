import numpy as np
import tensorflow as tf
import h5py

def convert_data(path='nyu_depth_v2_labeled.mat', save_to='./', test_ratio=0.2, seed=0):
    f = h5py.File(path)
    train_writer = tf.python_io.TFRecordWriter(save_to+'NYUv2_train.tfrecords')
    test_writer = tf.python_io.TFRecordWriter(save_to+'NYUv2_test.tfrecords')
    np.random.seed(seed)
    train_idces = np.random.choice(np.arange(f['images'].shape[0]), int((1-test_ratio)*f['images'].shape[0]),replace=False)
    for i, (image, depth) in enumerate(zip(f['images'], f['depths'])):
        image_raw = image.tostring()
        depth_raw = depth.tostring()
        example = tf.train.Example(
            features = tf.train.Features(
                feature = {
                    'depth': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[depth_raw])),
                    'image': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[image_raw])),
                    'width': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[image.shape[1]])),
                    'height': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[image.shape[2]]))}))
        if i in train_idces:
            train_writer.write(example.SerializeToString())
        else:
            test_writer.write(example.SerializeToString())

def load_data(traindata='./NYUv2_train.tfrecords', 
             testdata='./NYUv2_test.tfrecords', 
             batch_size=8, 
             seed=0):
    train_files = [traindata]
    test_files = [testdata]

    train_set = tf.data.TFRecordDataset(train_files).repeat()
    test_set = tf.data.TFRecordDataset(test_files).repeat()
    train_set = train_set.map(parser, num_parallel_calls=batch_size)
    test_set = test_set.map(parser, num_parallel_calls=batch_size)
    train_set = train_set.shuffle(buffer_size= 3 * batch_size, seed=seed)
    test_set = test_set.shuffle(buffer_size= 3 * batch_size, seed=seed)
    train_set = train_set.batch(batch_size)
    test_set = test_set.batch(batch_size)
    train_iter = train_set.make_one_shot_iterator()
    test_iter = test_set.make_one_shot_iterator()
    return train_iter.get_next(), test_iter.get_next()
    
def parser(serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'depth': tf.FixedLenFeature([], tf.string),
            'width': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64)
        })
    width = tf.cast(features['width'], tf.int32)
    height = tf.cast(features['height'], tf.int32)
    image = tf.transpose(tf.reshape(
        tf.decode_raw(features['image'], tf.uint8),
        [3, width, height]), perm=(2,1,0))
    depth = tf.transpose(tf.reshape(
        tf.decode_raw(features['depth'], tf.float32),
        [1, width, height]), perm=(2,1,0))

    return image, depth