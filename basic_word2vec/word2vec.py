# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 00:04:20 2017

@author: ADubey4
"""

import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import math
import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

os.chdir(r"C:\Users\Adubey4\Desktop\chatbot\code")

batch_size = 10000

words_df = pd.read_csv("word_pair_dataset_cw_2.csv", encoding='UTF-8')

_batch_index = 0;
#first_n_rows = words_df.iloc[:len(words_df.columns)]
le = LabelEncoder()
word_list = list(words_df.ix[:,0]) + list(words_df.ix[:,1])
le.fit(word_list)
#print(list(le.classes_[100:150]))

words_index_df = words_df.apply(lambda x: le.transform(x))
#le.classes_[words_index_df.iloc[0]]

with tf.name_scope("data"):
    center_words = tf.placeholder(tf.int32, shape=[batch_size], name = "center_words")
    target_words = tf.placeholder(tf.int32, shape=[batch_size,1], name = "target_words")

vocab_size = len(le.classes_)
embed_size = 100

with tf.name_scope("embed"):
    embed_matirx = tf.Variable(tf.random_uniform([vocab_size, embed_size], -1.0, 1.0),name="embed_matrix")

with tf.name_scope("loss"):
    embed = tf.nn.embedding_lookup(embed_matirx,center_words, name="embed")
    nce_weight = tf.Variable(tf.truncated_normal([vocab_size, embed_size], stddev = 1.0/ math.sqrt(embed_size)), name= "nce_weights")
    nce_bias = tf.Variable(tf.zeros([vocab_size]), name = "nce_bias")
    loss = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weight, biases= nce_bias, labels=target_words, inputs=embed, num_sampled=100, num_classes=vocab_size ),name="loss")
    
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


k = np.random.choice(len(words_df), 100, replace=False)
#b = ["king", "queen", "man", "women", "girl", "boy", "human", "humans"]
b = words_df.ix[k,0]
a = le.transform(b)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        if _batch_index+batch_size >= len(words_index_df):
            _batch_index = 0
        c_words = np.array(words_index_df.ix[:,0][_batch_index:_batch_index+batch_size])
        t_words = np.array(words_index_df.ix[:,1][_batch_index:_batch_index+batch_size]).reshape((batch_size,1))
        _, loss_val = sess.run([optimizer, loss], feed_dict = {center_words: c_words, target_words: t_words})
        print(loss_val)
        _batch_index = _batch_index + batch_size
#    e = embed_matirx.eval()[1:10]
    m_e = sess.run(tf.nn.embedding_lookup(embed_matirx, a, name="embed"))
#    m_e = sess.run([embed_matirx], feed_dict={center_words: a})


def plot_words(low_dim_targets, labels, filename='word_map.png'):
    plt.figure(figsize=(15, 15))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_targets[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)

tsne = PCA(n_components=2)
low_dim_targets = tsne.fit_transform(m_e)
labels = b
plot_words(low_dim_targets, labels)

#from sklearn.manifold import TSNE    
#tsne = TSNE(n_components=2, init='pca', n_iter=5000)