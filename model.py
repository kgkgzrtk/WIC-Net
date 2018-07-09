import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm, trange
from tensorflow.examples.tutorials.mnist import input_data

from ops import *


class wic_model():
    
    def __init__(self, sess, args):

        self.sess = sess
        self.model_name = args['--model_name']
        self.batch_size = int(args['--batch_size'])
        self.train_epoch = int(args['--epoch'])
        self.dim = int(args['--dim'])
        self.lr = float(args['--lr'])
        self.dataset_dir = args['--dataset_dir']
        self.tensorboard_dir = args['--tensorboard_dir']
        self.cp_dir = args['--checkpoint_dir']
        self.train_writer = tf.summary.FileWriter(self.tensorboard_dir + '/train', sess.graph)
        self.test_writer = tf.summary.FileWriter(self.tensorboard_dir + '/test', sess.graph)

        self.eps = 1e-8
        self.a_num = 10
        self.img_h, self.img_w, self.img_ch = (256, 256, 1)

    def build(self):

        self.x = tf.placeholder(tf.float32, [self.batch_size, 784])
        self.x_ = tf.image.resize_images(tf.reshape(self.x, [-1, 28, 28, 1]), [self.img_h, self.img_w])*2.-1.
        self.a = tf.placeholder(tf.float32, [self.batch_size, self.a_num])
        self.a_1h = tf.placeholder(tf.float32, [self.batch_size, self.a_num * 2])
        self.lmda_e = tf.placeholder(tf.float32, [1])

        self.y_enc = self.encoder(self.x_)
        self.x_dec = self.decoder(self.y_enc, self.a_1h)
        self.o_disc = self.discriminator(self.y_enc)

        self.gen_loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.x_, self.x_dec)))
        self.enc_loss = tf.reduce_mean(tf.reduce_sum(tf.log( tf.abs(self.o_disc - self.a) + self.eps ) ))
        self.disc_loss = - tf.reduce_mean(tf.reduce_sum(tf.log( tf.abs(self.o_disc - (1.- self.a)) + self.eps ) ))
        self.ae_loss = self.gen_loss - self.lmda_e * self.enc_loss

        optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5)
        train_vars = tf.trainable_variables()
        ae_vars = [v for v in train_vars if 'ae' in v.name]
        disc_vars = [v for v in train_vars if 'disc' in v.name]
        self.ae_optimizer = optimizer.minimize(self.ae_loss, var_list=ae_vars)
        self.disc_optimizer = optimizer.minimize(self.disc_loss, var_list=disc_vars)
        
        summary_dict = {
            'loss/gen': self.gen_loss,
            'loss/disc': self.disc_loss,
            'loss/ae':  self.ae_loss
        }

        self.update_met_list = []
        for k, v in summary_dict.items():
            mean_val, update_op = tf.contrib.metrics.streaming_mean(v, name=k)
            tf.summary.scalar(k, mean_val, collections=['train', 'test'])
            self.update_met_list.append(update_op)
        
        tf.summary.image('output_image', tf.cast((self.x_dec+1.)*127.5, tf.uint8), self.batch_size, collections=['train', 'test'])
        [tf.summary.histogram(v.name, v, collections=['train']) for v in train_vars if (('w' in v.name) or ('bn' in v.name))]

        self.train_merged = tf.summary.merge_all(key='train')
        self.test_merged = tf.summary.merge_all(key='test')
        self.merged = tf.summary.merge_all()

    def encoder(self, x, name='ae_enc'):
        with tf.variable_scope(name) as scope:
            c1 = conv(x, self.dim, name='c1')
            c2 = conv(c1, self.dim * 2, name='c2')
            c3 = conv(c2, self.dim * 4, name='c3')
            c4 = conv(c3, self.dim * 8, name='c4')
            c5 = conv(c4, self.dim * 16, name='c5')
            c6 = conv(c5, self.dim * 32, name='c6')
            c7 = conv(c6, self.dim * 32, name='c7')
            return c7
    
    def decoder(self, y, a, name='ae_dec'):
        with tf.variable_scope(name) as scope:
            a1 = tf.transpose(tf.stack([a]*4),[1, 0, 2])
            d0 = tf.concat([y, tf.reshape(a1, [-1, 2, 2, 2*self.a_num])], 3)
            d1 = deconv(d0, self.dim * 32, name='d1')
            
            a2 = tf.concat([a1]*4, axis=1)
            d1 = tf.concat([d1, tf.reshape(a2, [-1, 4, 4, 2*self.a_num])], 3)
            d2 = deconv(d1, self.dim * 16, name='d2')
            
            a3 = tf.concat([a2]*4, axis=1)
            d2 = tf.concat([d2, tf.reshape(a3, [-1, 8, 8, 2*self.a_num])], 3)
            d3 = deconv(d2, self.dim * 8, name='d3')
            
            a4 = tf.concat([a3]*4, axis=1)
            d3 = tf.concat([d3, tf.reshape(a4, [-1, 16, 16, 2*self.a_num])], 3)
            d4 = deconv(d3, self.dim * 4, name='d4')
            
            a5 = tf.concat([a4]*4, axis=1)
            d4 = tf.concat([d4, tf.reshape(a5, [-1, 32, 32, 2*self.a_num])], 3)
            d5 = deconv(d4, self.dim * 2, name='d5')
            
            a6 = tf.concat([a5]*4, axis=1)
            d5 = tf.concat([d5, tf.reshape(a6, [-1, 64, 64, 2*self.a_num])], 3)
            d6 = deconv(d5, self.dim * 1, name='d6')
            
            a7 = tf.concat([a6]*4, axis=1)
            d6 = tf.concat([d6, tf.reshape(a7, [-1, 128, 128, 2*self.a_num])], 3)
            d7 = deconv(d6, 3, name='d7')
            
            return tf.nn.tanh(d7)

    def discriminator(self, y, name='disc'):
        with tf.variable_scope(name) as scope:
            h1 = conv(y, self.dim * 32, name='h1')
            h1 = tf.nn.dropout(h1, keep_prob=0.7)
            h_flat = tf.reshape(h1, [self.batch_size, self.dim * 32])
            h2 = linear(h_flat, self.dim * 32, name='fc1')
            h3 = linear(h2, self.a_num, name='fc2')
            return tf.nn.sigmoid(h3)
    
    def trans_a(self, a):
        len_a = len(a)
        o = np.zeros([len_a, self.a_num * 2])
        for i in range(len_a):
            for j in range(self.a_num):
                o[i][2*j+int(a[i][j])] = 1
        return o

    def train(self):

        self.build()
        mnist = input_data.read_data_sets(self.dataset_dir, one_hot=True)

        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()
        per_epoch = mnist.train.labels.shape[0] // self.batch_size

        for epoch in trange(self.train_epoch, desc='epoch'):

            tf.local_variables_initializer().run()

            for i in trange(per_epoch):

                lmda_e = 0.0001 * (epoch * per_epoch + i) / (per_epoch * self.train_epoch)
                train_x, train_a = mnist.train.next_batch(self.batch_size)
                train_a_1h = self.trans_a(train_a)
                train_feed = {self.x: train_x, self.a: train_a, self.a_1h: train_a_1h, self.lmda_e: [lmda_e]}

                self.sess.run(self.disc_optimizer, feed_dict=train_feed)
                self.sess.run([self.ae_optimizer] + self.update_met_list, feed_dict=train_feed)

            train_summary = self.sess.run(self.train_merged, feed_dict = train_feed)
            self.train_writer.add_summary(train_summary, epoch)
            if epoch % 10 == 0: self.save_model(epoch)

    def save_model(self, epoch):
        file_name = self.model_name+"_e%05d.model" % epoch
        if not os.path.exists(self.cp_dir): os.makedirs(self.cp_dir)
        self.saver.save(self.sess, os.path.join(self.cp_dir, file_name))
