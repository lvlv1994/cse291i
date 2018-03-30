import tensorflow as tf

def subpix_conv2d(input, filters, kernel_size, scale=2, activation=tf.nn.leaky_relu, name=None):
    """
    Keras layer to do subpixel convolution.
    NOTE: Tensorflow backend only. Uses tf.depth_to_space
    Ref:
        [1] Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
            Shi et Al.
            https://arxiv.org/abs/1609.05158
    :param input: tensor shape, (batch, height, width, channel)
    :param scale: upsampling scale. Default=4
    :return:
    """
    X = tf.layers.conv2d(input, filters=filters*scale**2, kernel_size=kernel_size, padding='same', activation=activation)
    input_shape = X.get_shape().as_list()
    # upsample using depth_to_space
    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                input_shape[1] * scale,
                input_shape[2] * scale,
                int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.depth_to_space(x, scale)

    return tf.reshape(tensor=subpixel(X), shape=subpixel_shape(input_shape), name=name)
