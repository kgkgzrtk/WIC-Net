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
        self.z_dim = 1024
        self.img_h, self.img_w, self.img_ch = (256, 256, 3)
        self.layer_num = int(np.log2(self.img_h)) - 2

    def build(self):

        self.x = tf.placeholder(tf.float32, [self.batch_size, 256, 256, 3])
        self.a = tf.placeholder(tf.float32, [self.batch_size, self.a_num])
        self.a_1h = tf.placeholder(tf.float32, [self.batch_size, self.a_num * 2])
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim])
        self.training = tf.placeholder(tf.bool)

        self.x_ = tf.cond(self.training, lambda: self.data_augment(self.x), lambda: self.x)

        gen_o = self.generator(self.z, self.a_1h)
        self.x_dec = gen_o
        dis_o_real = self.discriminator(self.x_, self.a_1h)
        dis_o_fake = self.discriminator(gen_o, self.a_1h, reuse=True)
        
        #losses
        disc_loss = tf.reduce_mean(tf.reduce_sum(tf.math.softplus(-dis_o_real), axis=1)) \
                    + tf.reduce_mean(tf.reduce_sum(tf.math.softplus(dis_o_fake), axis=1))

        gen_loss = tf.reduce_mean(tf.reduce_sum(tf.math.softplus(-dis_o_fake), axis=1))

        optimizer = tf.train.AdamOptimizer(self.lr, beta1=0., beta2=0.9)
        train_vars = tf.trainable_variables()
        gen_vars = [v for v in train_vars if 'gen' in v.name]
        disc_vars = [v for v in train_vars if 'disc' in v.name]

        self.gen_optimizer = optimizer.minimize(gen_loss, var_list=gen_vars)
        self.disc_optimizer = optimizer.minimize(disc_loss, var_list=disc_vars)

        RMS_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(self.x_ - gen_o),[1, 2, 3])/(self.img_h * self.img_w)))
        
        summary_dict = {
            'loss/gen': gen_loss,
            'loss/disc': disc_loss,
            'eval/rms': RMS_loss, 
        }

        self.update_met_list = []
        for k, v in summary_dict.items():
            mean_val, update_op = tf.metrics.mean(v, name=k)
            tf.summary.scalar(k, mean_val, collections=['train', 'test'])
            self.update_met_list.append(update_op)
        
        self.x_img = tf.summary.image('x_image', tf.cast((self.x_+1.)*127.5, tf.uint8), self.batch_size, collections=['train', 'test'])
        a_1h_img = tf.reshape(tf.transpose(tf.stack([self.a_1h]*4),[1, 0, 2]), [self.batch_size, 2, 2*2*self.a_num, 1])
        self.a_1h_img = tf.summary.image('a_1h_img', tf.cast(a_1h_img*255., tf.uint8), self.batch_size, collections=['train', 'test'])
        self.sum_img = tf.summary.image('output_image', tf.cast((gen_o+1.)*127.5, tf.uint8), self.batch_size, collections=['train', 'test'])

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

    def res_block_up(self, x, a, o_dim, first_layer=False, final_layer=False, name='res_block_up'):
        with tf.variable_scope(name) as scope:
            if final_layer: o_dim = 3
            c_s = upsampling(x, x.shape[-1], name='up_s')
            c_s = conv(c_s, o_dim, c=1, k=1, name='skip_conv', func=False, bn=False)
            
            x = cond_batch_norm(x, a, name='cbn1')
            
            x = conv(x, o_dim, c=3, k=1, bn=False, name='conv1')
            
            x = cond_batch_norm(x, a, name='cbn2')
            x = upsampling(x, o_dim, name='up')

            x = conv(x, o_dim, c=3, k=1, name='conv2', func=final_layer, bn=False)
        return c_s + x

    def discriminator(self, x, a, name='disc', reuse=False):
        with tf.variable_scope(name, reuse=reuse) as scope:
            print("Discriminator:")
            half = self.layer_num // 2
            for i in range(half):
                x = self.res_block_down(x, self.dim * (2**i), name='b_dw1_'+str(i))
                print(x.shape)
            #self.attention(x, x.shape[-1], reuse=reuse)
            #print("self-attention x->x")
            for i in range(half, self.layer_num):
                x = self.res_block_down(x, self.dim * (2**i), name='b_dw2_'+str(i))
                print(x.shape)
            x = tf.nn.relu(x)
            x = tf.reduce_sum(x, axis=[1, 2])
            emb_a = embedding(a, self.a_num*2, x.shape[-1])
            emb = tf.reduce_sum(emb_a * x, axis=1, keepdims=True)
            o = emb + linear(x, 1)
        return x

    def generator(self, z, a, name='gen', reuse=False):
        with tf.variable_scope(name, reuse=reuse) as scope:
            print("Generator:")
            half = self.layer_num // 2
            ch = self.z_dim
            init_hw = 4
            z = linear(z, (init_hw**2)*ch, name="l_z")
            z = tf.reshape(z, [-1, init_hw, init_hw, ch])
            for i in range(half):
                z = self.res_block_up(z, a, ch//2,
                                    name='b_up1_'+str(i)
                                )
                ch = ch//2
                print(z.shape)
            #z = self.attention(z, z.shape[-1], reuse=reuse)
            #print("self-attention x->x")
            for i in range(half,self.layer_num):
                z = self.res_block_up(z, a, ch//2,
                                    final_layer=(i==self.layer_num-1),
                                    name='b_up2_'+str(i)
                                )
                ch = ch//2
                print(z.shape)
            return tf.nn.tanh(z)

    def attention(self, x, ch, sn=True, name='attention', reuse=False):
        with tf.variable_scope(name, reuse=reuse) as scope: 
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
                train_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                train_feed = {self.x: train_x, self.a: train_a, self.a_1h: train_a_1h, self.z: train_z, self.training: True}
                #self.sess.run(self.crit_optimizer, feed_dict=train_feed)
                self.sess.run(self.disc_optimizer, feed_dict=train_feed)
                self.sess.run([self.gen_optimizer] + self.update_met_list, feed_dict=train_feed)

            train_x = self.train_image[:self.batch_size]
            train_a = self.train_attr[:self.batch_size]
            train_a_1h = self.trans_a(train_a)
            train_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
            train_feed = {self.x: train_x, self.a: train_a, self.a_1h: train_a_1h,  self.z: train_z, self.training: False}
            train_summary = self.sess.run(self.train_merged, feed_dict = train_feed)
            self.train_writer.add_summary(train_summary, epoch)
            
            for i in range(per_epoch_):
                batch = self.batch_size * i
                test_x = self.test_image[batch:batch+self.batch_size]
                test_a = self.test_attr[batch:batch+self.batch_size]
                test_a_1h = self.trans_a(test_a)
                test_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                test_feed = {self.x: test_x, self.a: test_a, self.a_1h: test_a_1h, self.z: test_z, self.training: False}
                self.sess.run(self.update_met_list, feed_dict=test_feed)
            
            disp_x = self.test_image[:self.batch_size]
            disp_a = self.test_attr[:self.batch_size]
            disp_a_1h = self.trans_a(disp_a)
            disp_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
            disp_feed = {self.x: disp_x, self.a: disp_a, self.a_1h: disp_a_1h, self.z: disp_z, self.training: False}
            disp_summary = self.sess.run(self.test_merged, feed_dict=disp_feed)
            self.test_writer.add_summary(disp_summary, epoch)
                
            if epoch % 10 == 0: self.save_model(epoch)
    
    def weather_run(self, images, attrs, ind, save=False):
        a_swap_li = self.swap_attr(attrs, ind)
        swap_img_li = []
        for a_s in tqdm(a_swap_li):
            a_s_1h = self.trans_a(attrs)
            swap_feed = {self.x: images, self.a: a_s, self.a_1h: a_s_1h, self.lmda_e: [0], self.training: False}
            swap_img = self.sess.run(self.x_dec, feed_dict=swap_feed)
            swap_img_li.append((swap_img+1.)*127.5)
        swapped = np.resize(np.transpose(swap_img_li, [1, 2, 0, 3, 4]), [256*16, 256*11, 3])
        if save:
            swapped_img = Image.fromarray(swapped.astype('uint8'))
            swapped_img.save('./results/swapped/swap_%s_%d.jpg'%(self.load_name, ind))
        return swapped

    def swap_attr(self, attrs, ind):
        new_attrs = np.array(attrs)
        attrs[:,ind] = 0.
        new_attrs[:,ind] = 1.
        attr_li = [(1.-i)*attrs + i*new_attrs for i in np.arange(0.0, 1.1, 0.1)]
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
