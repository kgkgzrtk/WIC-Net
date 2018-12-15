import tensorflow as tf
import numpy as np

def batch_norm(x, name='bn'):
    with tf.variable_scope(name) as scope:
        eps = 1e-5
        shape = x.get_shape().dims[-1].value
        gamma = tf.get_variable('gamma', [shape], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [shape], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(x, axes=[0, 1, 2], keep_dims=False)
        return tf.nn.batch_normalization(x, mean, variance, beta, gamma, variance_epsilon=eps)

def cond_batch_norm(x, c=None, hidden_size=64, name='cbn'):
    with tf.variable_scope(name) as scope:
        eps = 1e-5
        batch, hei, wid, ch = x.shape.as_list()
        gamma = tf.get_variable('gamma', [batch, ch], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [batch, ch], initializer=tf.constant_initializer(0.0))
        if c is not None:
            d_g = cbn_func(c, hidden_size, ch, name='f_gamma')
            d_b = cbn_func(c, hidden_size, ch, name='f_beta')
            gamma_ = gamma + d_g
            beta_ = beta + d_b
            gamma__ = tf.transpose(tf.stack([gamma_]*hei), [1, 0, 2])
            gamma__ = tf.transpose(tf.stack([gamma__]*wid), [1, 2, 0, 3])
            beta__ = tf.transpose(tf.stack([beta_]*hei), [1, 0, 2])
            beta__ = tf.transpose(tf.stack([beta__]*wid), [1, 2, 0, 3])
        else:
            gamma__ = gamma
            beta__ = beta
        mean, variance = tf.nn.moments(x, axes=[0, 1, 2], keep_dims=False)
        sigma = tf.math.sqrt(variance + eps)
        x_norm = (x - mean) / sigma
        out = gamma__ * x_norm + beta__
        return out

def cbn_func(x, n_dim, f_num, name="cbn_func"):
    with tf.variable_scope(name) as scope:
        x = linear(x, n_dim, name='l1')
        x = tf.nn.relu(x, name='relu')
        x = linear(x, f_num, name='l2')
    return x

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name) as scope:
        return tf.maximum(x, leak*x)

def swish(x, name="swish"):
    with tf.variable_scope(name) as scope:
        return x*tf.sigmoid(x)

def linear(input_, output_size, sn=True, name='linear_layer'):
    with tf.variable_scope(name) as scope:
        input_ = tf.layers.flatten(input_)
        shape = input_.shape[-1]
        stddev = np.sqrt(1./output_size)
        W = tf.get_variable('w', [shape, output_size], initializer=tf.truncated_normal_initializer(0.0, stddev))
        if sn: W = spec_norm(W)
        b = tf.get_variable('b', [output_size], initializer=tf.constant_initializer(0.0))
        return tf.matmul(input_, W) + b

def conv(x, out_dim, name='Conv', c=3, k=2, stddev=0.02, padding='SAME', bn=True, sn=True, func=False, func_factor=0.0):
    with tf.variable_scope(name) as scope:
        W = tf.get_variable('w', [c, c, x.get_shape().dims[-1].value, out_dim], initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [out_dim], initializer=tf.constant_initializer(0.0))

        if sn: filter_ = spec_norm(W)
        else: filter_ = W

        y = tf.nn.conv2d(x, filter_, strides=[1, k, k, 1], padding=padding) + b

        if bn: 
            y = tf.contrib.layers.batch_norm(y, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope='bn')
        if func:
            if func_factor != 0:
                y = lrelu(y, leak=func_factor)
            else:
                y = tf.nn.relu(y, name='relu')
        return y

def deconv(x, out_dim, name='Deconv', c=4, k=2, stddev=0.02, padding='SAME', bn=True, sn=True, func=True, func_factor=0.0):
    with tf.variable_scope(name) as scope:
        x_shape = x.get_shape().as_list()
        out_shape = [x_shape[0], x_shape[1]*k, x_shape[2]*k, out_dim]
        W = tf.get_variable('w', [c, c, out_dim, x.get_shape()[-1]], initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [out_dim], initializer=tf.constant_initializer(0.0))

        if sn: filter_ = spec_norm(W)
        else: filter_ = W

        y = tf.nn.conv2d_transpose(x, filter_, out_shape, strides=[1, k, k, 1], padding=padding)
            
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

def upsampling(x, out_dim, scale=2, name='resize'):
    return tf.image.resize_bilinear(x, [x.shape[1]*scale, x.shape[2]*scale])

def resize_conv(x, out_dim, name='resize_Conv', c=4, k=1, bn=True, func=True):
        x = tf.image.resize_bilinear(x, [x.shape[1]*2, x.shape[2]*2])
        y = conv(x, out_dim, c=c, k=k, name=name, bn=bn, padding='SAME', func=func, func_factor=0)
        return y

def pixel_shuffler(x, out_shape, r=2, c=4, name='ps', bn=True):
    with tf.variable_scope(name) as scope:
        y_conv = conv(x, out_shape[3]*(r**2), c=c, k=1, bn=bn)
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

def hw_flatten(x):
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

def batch_dot(a, b):
    a = tf.expand_dims(a, 1)
    b = tf.expand_dims(b, 2)
    return tf.matmul(a,b)

def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v**2)**0.5+eps)

def embedding(y, in_size, out_size):
    V = tf.get_variable('v', [in_size, out_size], initializer=tf.truncated_normal_initializer(stddev=1.0))
    V = spec_norm(V)
    o = tf.matmul(y, V)
    return o
    
def spec_norm(w):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
    u_hat = u
    v_ = tf.matmul(u_hat, tf.transpose(w))
    v_hat = l2_norm(v_)
    u_ = tf.matmul(v_hat, w)
    u_hat = l2_norm(u_)
    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w/sigma
    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)
    return w_norm
