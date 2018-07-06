import tensorflow as tf
import numpy as np

def batch_norm(input_, name='bn'):
    with tf.variable_scope(name) as scope:
        eps = 1e-5
        shape = input_.get_shape().dims[3].value
        gamma = tf.get_variable('gamma', [shape], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [shape], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input_, axes=[0, 1, 2], keep_dims=False)
        return tf.nn.batch_normalization(input_, mean, variance, beta, gamma, variance_epsilon=eps)

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name) as scope:
        return tf.maximum(x, leak*x)

def swish(x, name="swish"):
    with tf.variable_scope(name) as scope:
        return x*tf.sigmoid(x)

def linear(input_, output_size, name='linear_layer'):
    with tf.variable_scope(name) as scope:
        shape = input_.shape[-1]
        stddev = np.sqrt(1./output_size)
        W = tf.get_variable('w', [shape, output_size], initializer=tf.truncated_normal_initializer(0.0, stddev))
        b = tf.get_variable('b', [output_size], initializer=tf.constant_initializer(0.0))
        return tf.matmul(input_, W) + b

def conv(image, out_dim, name='Conv', c=4, k=2, stddev=0.02, padding='SAME', bn=True, func=False, func_factor=0.2):
    with tf.variable_scope(name) as scope:
        W = tf.get_variable('w', [c, c, image.get_shape().dims[-1].value, out_dim], initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [out_dim], initializer=tf.constant_initializer(0.0))
        y = tf.nn.conv2d(image, W, strides=[1, k, k, 1], padding=padding) + b
        if bn: 
            y = tf.contrib.layers.batch_norm(y, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope='bn')
        if func:
            if func_factor != 0:
                y = lrelu(y, leak=func_factor)
            else:
                y = tf.nn.relu(y, name='relu')
        return y

def deconv(image, out_dim, name='Deconv', c=4, k=2, stddev=0.02, padding='SAME', bn=True, func=False, func_factor=0.2):
    with tf.variable_scope(name) as scope:
        y = tf.contrib.layers.conv2d_transpose(image, out_dim, [c, c], [k, k], padding,
                activation_fn=None,
                weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                )
        if bn:
            y = tf.contrib.layers.batch_norm(y, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope='bn')
        if func:
            if func_factor != 0:
                y = lrelu(y, leak=func_factor)
            else:
                y = tf.nn.relu(y, name='relu')
        return y

def gaussian_noise_layer(x, std=0.05):
    noise = tf.random_normal(shape=tf.shape(x), mean=0.0, stddev=std) 
    return x + noise

def resize_conv(image, output_shape, name, c=4, k=1, bn=True):
    image = tf.image.resize_bilinear(image, [output_shape[1], output_shape[2]])
    y = conv(image, output_shape[-1], name=name, c=c, k=k, bn=bn, padding='SAME')
    return y

def pixel_shuffler(image, out_shape, r=2, c=4, name='ps', bn=True):
    with tf.variable_scope(name) as scope:
        y_conv = conv(image, out_shape[3]*(r**2), c=c, k=1, bn=bn)
        y_list = tf.split(y_conv, out_shape[3], 3)
        pix_map_list = []
        for y in y_list:
            b, h, w, c = y.get_shape().as_list()
            pix_map = tf.reshape(y, [b, h, w, r, r])
            pix_map = tf.transpose(pix_map, perm=[0, 1, 2, 4, 3])
            pix_map = tf.split(pix_map,h,1)
            pix_map = tf.concat([tf.squeeze(m,1) for m in pix_map],2)
            pix_map = tf.split(pix_map,w,1)
            pix_map = tf.concat([tf.squeeze(m,1) for m in pix_map],2)
            pix_map = tf.reshape(pix_map, [b, h*r, w*r, 1])
            pix_map_list.append(pix_map)
        out = tf.concat(pix_map_list, 3)
        return out
