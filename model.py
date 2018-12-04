import tensorflow as tf
import numpy as np
import pandas as pd
import os
import glob

from PIL import Image
from tqdm import tqdm, trange
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime

from ops import *
from data_io import *


class wic_model():
    
    def __init__(self, sess, args):

        self.sess = sess
        self.model_name = args['--model_name']
        self.load_name = args['--load_name']
        self.batch_size = int(args['--batch_size'])
        self.train_epoch = int(args['--epoch'])
        self.dim = int(args['--dim'])
        self.lr = float(args['--lr'])
        self.dataset_dir = args['--dataset_dir']
        self.tensorboard_dir = args['--tensorboard_dir']
        self.cp_dir = args['--checkpoint_dir']
        self.max_lmda = float(args['--max_lmda'])
        self.reg_scale = float(args['--reg_scale'])

        self.train_writer = tf.summary.FileWriter(self.tensorboard_dir + '/train', sess.graph)
        self.test_writer = tf.summary.FileWriter(self.tensorboard_dir + '/test', sess.graph)

        self.eps = 1e-8
        self.a_num = 6
        self.img_h, self.img_w, self.img_ch = (256, 256, 3)

    def build(self):

        self.x = tf.placeholder(tf.float32, [self.batch_size, 256, 256, 3])
        self.a = tf.placeholder(tf.float32, [self.batch_size, self.a_num])
        self.a_1h = tf.placeholder(tf.float32, [self.batch_size, self.a_num * 2])
        self.lmda_e = tf.placeholder(tf.float32, [1])
        self.training = tf.placeholder(tf.bool)

        self.x_ = tf.cond(self.training, lambda: self.data_augment(self.x), lambda: self.x)

        self.y_enc = self.encoder(self.x_)
        self.x_dec = self.decoder(self.y_enc, self.a_1h)
        self.o_disc = self.discriminator(self.y_enc)
        
        #losses
        self.gen_loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.x_, self.x_dec), axis=[1, 2, 3]))
        self.enc_loss = tf.reduce_mean(tf.reduce_sum(tf.log( tf.abs(self.o_disc - self.a) + self.eps ), axis=1))
        self.disc_loss = - tf.reduce_mean(tf.reduce_sum(tf.log( tf.abs(self.o_disc - (1.- self.a)) + self.eps ), axis=1))
        self.ae_loss = self.gen_loss - self.lmda_e * self.enc_loss

        optimizer = tf.train.AdamOptimizer(self.lr, beta1=0., beta2=0.9)
        train_vars = tf.trainable_variables()
        ae_vars = [v for v in train_vars if 'ae' in v.name]
        disc_vars = [v for v in train_vars if 'disc' in v.name]

        self.L1_weight_penalty = tf.add_n([tf.reduce_sum(tf.abs(w)) for w in ae_vars if 'w' in w.name])
        self.weight_penalty = self.reg_scale * self.L1_weight_penalty

        self.ae_optimizer = optimizer.minimize(self.ae_loss + self.weight_penalty, var_list=ae_vars)
        self.disc_optimizer = optimizer.minimize(self.disc_loss, var_list=disc_vars)

        RMS_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(self.x_ - self.x_dec),[1, 2, 3])/(self.img_h * self.img_w)))
        
        summary_dict = {
            'loss/gen': self.gen_loss,
            'loss/disc': self.disc_loss,
            'loss/ae':  self.ae_loss,
            'loss/rms': RMS_loss, 
        }

        self.update_met_list = []
        for k, v in summary_dict.items():
            mean_val, update_op = tf.metrics.mean(v, name=k)
            tf.summary.scalar(k, mean_val, collections=['train', 'test'])
            self.update_met_list.append(update_op)
        
        self.x_img = tf.summary.image('x_image', tf.cast((self.x_+1.)*127.5, tf.uint8), self.batch_size, collections=['train', 'test'])
        a_1h_img = tf.reshape(tf.transpose(tf.stack([self.a_1h]*4),[1, 0, 2]), [self.batch_size, 2, 2*2*self.a_num, 1])
        self.a_1h_img = tf.summary.image('a_1h_img', tf.cast(a_1h_img*255., tf.uint8), self.batch_size, collections=['train', 'test'])
        self.sum_img = tf.summary.image('output_image', tf.cast((self.x_dec+1.)*127.5, tf.uint8), self.batch_size, collections=['train', 'test'])

        [tf.summary.histogram(v.name, v, collections=['train']) for v in train_vars if (('w' in v.name) or ('bn' in v.name))]

        self.train_merged = tf.summary.merge_all(key='train')
        self.test_merged = tf.summary.merge_all(key='test')
        self.merged = tf.summary.merge_all()

    def res_block_down(self, x, o_dim, name='res_block_down'):
        with tf.variable_scope(name) as scope:
            c_s = conv(x, o_dim, c=1, k=1, name='skip_conv', func=False, bn=False)
            c_s = tf.layers.average_pooling2d(c_s, 2, 2, padding='same', name='avrpoo')
            c1 = conv(x, o_dim, c=3, k=1, name='conv1', func_factor=0)
            c2 = conv(c1, o_dim, c=3, k=2, name='conv2', func_factor=0)
        return c_s + c2

    def res_block_up(self, x, a, o_dim, name='res_block_up'):
        with tf.variable_scope(name) as scope:
            c_s = conv(x, o_dim, c=1, k=1, name='skip_conv', func=False, bn=False)
            c_s = upsampling(c_s, o_dim, name='up')
            c1 = conv(x, o_dim, c=3, k=1, name='conv1', func_factor=0)
            c2 = conv(c1, o_dim, c=3, k=2, name='conv2', func_factor=0)
        return c_s + c2



    def encoder(self, x, name='ae_enc'):
        with tf.variable_scope(name) as scope:
            c1 = conv(x, self.dim, name='c1')
            c2 = conv(c1, self.dim * 2, name='c2')
            c3 = conv(c2, self.dim * 4, name='c3')
            c3 = self.attention(c3, self.dim * 4)

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
            d1 = tf.nn.dropout(d1, keep_prob=0.7)
            
            a2 = tf.concat([a1]*4, axis=1)
            d1 = tf.concat([d1, tf.reshape(a2, [-1, 4, 4, 2*self.a_num])], 3)
            d2 = deconv(d1, self.dim * 16, name='d2')
            d2 = tf.nn.dropout(d2, keep_prob=0.7)
            
            a3 = tf.concat([a2]*4, axis=1)
            d2 = tf.concat([d2, tf.reshape(a3, [-1, 8, 8, 2*self.a_num])], 3)
            d3 = deconv(d2, self.dim * 8, name='d3')

            d3 = self.attention(d3, self.dim*8)
            
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
            d7 = deconv(d6, 3, name='d7', func=False, bn=False)
            
            return tf.nn.tanh(d7)

    def discriminator(self, y, name='disc'):
        with tf.variable_scope(name) as scope:
            h1 = conv(y, self.dim * 32, name='h1')
            h1 = tf.nn.dropout(h1, keep_prob=0.7)
            h_flat = tf.reshape(h1, [self.batch_size, self.dim * 32])
            h2 = linear(h_flat, self.dim * 32, name='fc1')
            h3 = linear(h2, self.a_num, name='fc2')
            return tf.nn.sigmoid(h3)


    def attention(self, x, ch, sn=True, name='attention', reuse=False):
        with tf.variable_scope(name) as scope: 
            f = conv(x, ch // 8, c=1, k=1, bn=False, sn=sn, name='f_conv')
            g = conv(x, ch // 8, c=1, k=1, bn=False, sn=sn, name='g_conv')
            h = conv(x, ch, c=1, k=1, bn=False, sn=sn, name='h_conv')
            s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)
            beta = tf.nn.softmax(s, axis=-1)
            o = tf.matmul(beta, hw_flatten(h))
            gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
            o = tf.reshape(o, shape=x.shape)
            x = gamma*o + x
        return x

    def trans_a(self, a):
        len_a = len(a)
        o = np.zeros([len_a, self.a_num * 2])
        for i in range(len_a):
            for j in range(self.a_num):
                o[i][2*j] = 1.-a[i][j]
                o[i][2*j+1] = a[i][j]
        return o

    def swap_col(self, gray_images):
        b_arr = [np.unpackbits(x)[-3:] for x in np.arange(1,8, dtype=np.uint8)]
        label_arr = np.eye(len(b_arr))
        col_images = []
        labels = []
        for img in gray_images:
            ind = np.random.randint(len(b_arr))
            col_images.append(b_arr[ind] * np.stack((img,)*3, -1))
            labels.append(label_arr[ind].astype(np.float32))
        return col_images, labels

    def train(self):
        self.load_data()

        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()
        per_epoch = len(self.train_image) // self.batch_size
        per_epoch_ = len(self.test_image) // self.batch_size

        for epoch in trange(self.train_epoch, desc='epoch'):

            tf.local_variables_initializer().run()
            perm = np.random.permutation(len(self.train_image))

            for i in trange(per_epoch, desc='iter'):
                batch = self.batch_size * i

                lmda_e = self.max_lmda * (epoch * per_epoch + i) / (per_epoch * self.train_epoch)
                train_x = self.train_image[perm[batch:batch+self.batch_size]]
                train_a = self.train_attr[perm[batch:batch+self.batch_size]]
                train_a_1h = self.trans_a(train_a)
                train_feed = {self.x: train_x, self.a: train_a, self.a_1h: train_a_1h, self.lmda_e: [lmda_e], self.training: True}
                #self.sess.run(self.crit_optimizer, feed_dict=train_feed)
                self.sess.run(self.disc_optimizer, feed_dict=train_feed)
                self.sess.run([self.ae_optimizer] + self.update_met_list, feed_dict=train_feed)

            train_x = self.train_image[:self.batch_size]
            train_a = self.train_attr[:self.batch_size]
            train_a_1h = self.trans_a(train_a)
            train_feed = {self.x: train_x, self.a: train_a, self.a_1h: train_a_1h, self.lmda_e: [0], self.training: False}
            train_summary = self.sess.run(self.train_merged, feed_dict = train_feed)
            self.train_writer.add_summary(train_summary, epoch)
            
            for i in range(per_epoch_):
                batch = self.batch_size * i
                test_x = self.test_image[batch:batch+self.batch_size]
                test_a = self.test_attr[batch:batch+self.batch_size]
                test_a_1h = self.trans_a(test_a)
                test_feed = {self.x: test_x, self.a: test_a, self.a_1h: test_a_1h, self.lmda_e: [0], self.training: False}
                self.sess.run(self.update_met_list, feed_dict=test_feed)
            
            disp_x = self.test_image[:self.batch_size]
            disp_a = self.test_attr[:self.batch_size]
            disp_a_1h = self.trans_a(disp_a)
            self.weather_run(disp_x, disp_a, 0)
            disp_feed = {self.x: disp_x, self.a: disp_a, self.a_1h: disp_a_1h, self.lmda_e: [0], self.training: False}
            disp_summary = self.sess.run(self.test_merged, feed_dict=disp_feed)
            self.test_writer.add_summary(disp_summary, epoch)
                
            if epoch % 10 == 0: self.save_model(epoch)
    
    def weather_run(self, images, attrs, ind):
        a_swap_li = self.swap_attr(attrs, ind)
        swap_img_li = []
        for a_s in tqdm(a_swap_li):
            a_s_1h = self.trans_a(attrs)
            swap_feed = {self.x: images, self.a: a_s, self.a_1h: a_s_1h, self.lmda_e: [0], self.training: False}
            swap_img = self.sess.run(self.x_dec, feed_dict=swap_feed)
            swap_img_li.append((swap_img+1.)*127.5)
        swapped = np.resize(np.transpose(swap_img_li, [1, 2, 0, 3, 4]), [256*16, 256*11, 3])
        swapped_img = Image.fromarray(swapped.astype('uint8'))
        swapped_img.save('./results/swapped/swap_%s_%d.jpg'%(self.load_name, ind))

    def swap_attr(self, attrs, ind):
        new_attrs = np.array(attrs)
        #attrs[:,ind] = 0.
        #new_attrs[:,ind] = 1.
        #attr_li = [(1.-i)*attrs + i*new_attrs for i in np.arange(0.0, 1.1, 0.1)]
        attr_li = [(1.-i)*attrs + i*(1.-attrs) for i in np.arange(0.0, 1.1, 0.1)]
        return attr_li

    def save_model(self, epoch):
        file_name = self.model_name+"_e%05d.model" % epoch
        if not os.path.exists(self.cp_dir): os.makedirs(self.cp_dir)
        self.saver.save(self.sess, os.path.join(self.cp_dir, file_name))

    def load_model(self):
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, os.path.join(self.cp_dir, self.load_name))

    def load_data(self):
        cols = ['id', 'loc.lat', 'loc.lng', 'weather.date.hour', 'weather.tempm', 'weather.hum', 'weather.wgustm']
        df = load_json_file(cols)
        df[cols[1:]] = self.normalize(df[cols[1:]])
        image_li = []
        attr_li = []
        for ind, row in tqdm(df.iterrows()):
            #load image
            fn = str(row['id'])
            file_path = '/mnt/fs2/2018/matsuzaki/dataset_fromnitta/Image/*/' + fn + '.jpg'
            files = glob.glob(file_path)
            try:
                img = Image.open(files[0], 'r')
            except:
                print('not found '+fn+'.jpg')
                continue
            img = img.resize((256,256))
            #load sensor vals
            attr = row[1:].values

            image_li.append(np.asarray(img)/255.*2.-1.)
            attr_li.append(attr)
        data_li = list(zip(image_li, attr_li))
        train_li = [v for i, v in enumerate(data_li) if i%10!=0]
        test_li = [v for i, v in enumerate(data_li) if i%10==0]
        (train_image, train_attr) = list(zip(*train_li))
        (test_image, test_attr) = list(zip(*test_li))
        self.train_image = np.asarray(train_image)
        self.train_attr = np.asarray(train_attr)
        self.test_image = np.asarray(test_image)
        self.test_attr = np.asarray(test_attr)

    def normalize(self, df):
        df_ = df.copy()
        for c_name in df.columns:
            max_ = df[c_name].max()
            min_ = df[c_name].min()
            df_[c_name] = (df[c_name] - min_)/(max_ - min_)
        return df_

    def data_augment(self, imgs):
        imgs_ = []
        for img in tf.split(imgs, self.batch_size):
            img = tf.squeeze(img)
            img = tf.image.random_flip_left_right(img)
            rotation = 10.
            rnd_theta = np.random.uniform(-rotation*(np.pi/180.), rotation*(np.pi/180.))
            img = tf.contrib.image.rotate(img, rnd_theta)
            abs_theta = np.absolute(rnd_theta)
            crop_size_h = int(self.img_h * ((np.cos(abs_theta)/(np.sin(abs_theta)+np.cos(abs_theta)))**2))
            crop_size_w = int(crop_size_h*(self.img_w/self.img_h))
            img = tf.image.resize_image_with_crop_or_pad(img, crop_size_h, crop_size_w)

            #Random crop
            rnd_scale = np.random.uniform(1.0, 1.5)
            img = tf.image.resize_images(img, [int(self.img_h*rnd_scale), int(self.img_w*rnd_scale)])
            img = tf.random_crop(img, [self.img_h, self.img_w, 3])

            #Random brightness & contrast
            img = tf.image.random_brightness(img, max_delta=0.4)
            img = tf.image.random_contrast(img, lower=0.9, upper=1.1)

            imgs_.append(img)

        return tf.stack(imgs_)
    
    #gradient penalty https://arxiv.org/pdf/1704.00028
    def gradient_penalty(self):
        eps = tf.random_uniform([self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        x_hat = eps * self.x_dec + (1. - eps) * self.x_
        o_crit_hat = self.critic(x_hat, reuse=True)[-1]
        ddy = tf.gradients(o_crit_hat, [x_hat])[0]
        ddy = tf.sqrt(tf.reduce_sum(tf.square(ddy), [1, 2, 3]))
        return tf.reduce_mean(tf.square(ddy - 1.))
