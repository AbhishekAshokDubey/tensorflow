# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 12:56:09 2017

@author: ADubey4
"""

import os
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import math

import tensorflow as tf

def normalize(X):
    mean = np.mean(X,axis=0)
    std = np.std(X,axis=0)
    X_norm = (X-mean)/std
    X_norm[np.isnan(X_norm) | np.isinf(X_norm)] = 0
    return X_norm, mean, std

base_folder_path = r"C:\Users\Adubey4\Desktop\tf\GAN101"
os.chdir(base_folder_path)

all_raw_images_np = np.load(os.path.join(base_folder_path, "data",'cifar_all_images.npy'))
#all_raw_images_np = np.savetxt(os.path.join(base_folder_path, "data",'cifar_all_images.txt'))
#all_raw_images_np = all_raw_images_np[1:100000]
all_raw_images_np, mean, std =  normalize(all_raw_images_np)


# for real data
image_placeholder = tf.placeholder(tf.float32, shape=[None, 3072], name='image_batch')
# discriminator
D_W1 = tf.Variable(tf.truncated_normal([3072, 300],stddev=1.0 / math.sqrt(float(3074))), name='D_W1')
D_b1 = tf.Variable(tf.zeros(shape=[300]), name='D_b1')
D_W2 = tf.Variable(tf.truncated_normal([300, 1], stddev=1.0 / math.sqrt(float(3074))), name='D_W2')
D_b2 = tf.Variable(tf.zeros(shape=[1]), name='D_b2')
D_param_list = [D_W1, D_W2, D_b1, D_b2]

# for fake data
noise_placeholder = tf.placeholder(tf.float32, shape=[None, 100], name='Z')
# generator
G_W1 = tf.Variable(tf.truncated_normal([100,700],stddev=1.0 / math.sqrt(float(3074))), name='G_W1')
G_b1 = tf.Variable(tf.zeros(shape=[700]), name='G_b1')
G_W2 = tf.Variable(tf.truncated_normal([700,3072],stddev=1.0 / math.sqrt(float(3074))), name='G_W2')
G_b2 = tf.Variable(tf.zeros(shape=[3072]), name='G_b2')
G_param_list = [G_W1, G_W2, G_b1, G_b2]

def g_net(Z):
    G_h1 = tf.nn.relu(tf.matmul(Z, G_W1) + G_b1)
    return tf.nn.sigmoid(tf.matmul(G_h1, G_W2) + G_b2)

def d_net(X):
    D_h1 = tf.nn.relu(tf.matmul(X, D_W1) + D_b1)
    return tf.nn.sigmoid(tf.matmul(D_h1, D_W2) + D_b2)

def get_noise(m,n):
    return np.random.uniform(-1.0, 1.0, size=[m, n])

def get_batch_indx(sample_len = 0, data_len = 1, start_indx = 0, random=False):
    if random:
        assert data_len>sample_len, "data_len should be greater than sample_len"
        return np.random.choice(range(0, data_len), sample_len, replace=False)
    else:
        return np.array(range(start_indx, start_indx+sample_len))


fake_sample = g_net(noise_placeholder)

real_label = d_net(image_placeholder)
fake_label = d_net(fake_sample)

D_loss = -tf.reduce_mean(tf.log(real_label) + tf.log(1.0 - fake_label))
G_loss = -tf.reduce_mean(tf.log(fake_label))

D_opt = tf.train.AdamOptimizer().minimize(D_loss, var_list=D_param_list)
G_opt = tf.train.AdamOptimizer().minimize(G_loss, var_list=G_param_list)

epochs = 3
batch_size = 50
data_len = len(all_raw_images_np)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch_no in range(epochs):
        for batch_no in range(int(data_len/batch_size)):
            indx = get_batch_indx(sample_len=batch_size, start_indx=batch_no*batch_size)
            _, D_loss_curr = sess.run([D_opt, D_loss], feed_dict={image_placeholder: all_raw_images_np[indx], noise_placeholder: get_noise(batch_size, 100)})
            _, G_loss_curr = sess.run([G_opt, G_loss], feed_dict={noise_placeholder: get_noise(batch_size, 100)})
            print(D_loss_curr, G_loss_curr)
    generated_img_samples = sess.run(fake_sample, feed_dict={noise_placeholder: get_noise(30, 100)})

# de-normalise
generated_img_samples = (generated_img_samples*std) + mean
# reshape and read as uint8 (0,255)
batch_images_display = generated_img_samples.reshape(len(generated_img_samples), 3, 32, 32).transpose(0,2,3,1).astype("uint8")

fig, axes1 = plt.subplots(4,4,figsize=(3,3))
for j in range(4):
    for k in range(4):
        i = np.random.choice(range(len(batch_images_display)))
        axes1[j][k].set_axis_off()
        axes1[j][k].imshow(batch_images_display[i:i+1][0])