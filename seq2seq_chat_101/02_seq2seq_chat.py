# -*- coding: utf-8 -*-
"""
Created on Sun May 14 20:28:22 2017

@author: ADubey4
"""
import numpy as np
import pandas as pd
import re
from collections import Counter
import tensorflow as tf

## tf >= 1.1
## conda install -c conda-forge tensorflow=1.1
## pip install tensorflow
#from tensorflow.contrib.legacy_seq2seq import embedding_attention_seq2seq
#from tensorflow.contrib.rnn import BasicLSTMCell, MultiRNNCell
#from tensorflow.contrib.legacy_seq2seq import sequence_loss_by_example

# tf < 1.0
# conda install -c conda-forge tensorflow=0.12
# pip > tensorflow=0.12
embedding_attention_seq2seq = tf.nn.seq2seq.embedding_attention_seq2seq
BasicLSTMCell, MultiRNNCell = tf.nn.rnn_cell.BasicLSTMCell, tf.nn.rnn_cell.MultiRNNCell
sequence_loss_by_example = tf.nn.seq2seq.sequence_loss


from data_util import *
#import os
#base_folder_path = os.path.dirname(__file__);
#os.chdir(base_folder_path);
#session = tf.InteractiveSession()

raw_data_file = pd.read_csv("cornell_raw.csv")
raw_data_file = raw_data_file.ix[1:10000,:]
index2word_list = ["PAD","GO","EOF", "NDC"]
word2index_dict = {}

reverse_encoder_input = True

# Training or Predicting
train = False
#train = True

# Training parameters
batch_size = 10;
epochs = 1;

# embedding & seq2seq layers info
embeding_size = 300
num_layers = 1;

index2word_list = index2word_list + get_index2word_list(raw_data_file.copy())
word2index_dict = get_word2index_dict(index2word_list)
encoder_inputs_list, decoder_labels_list = get_encoder_deconder_inputs(raw_data_file.copy(),
                                                                       word2index_dict)
encoder_inputs_np, decoder_labels_np, max_encoder_length, max_decoder_length = align_encoder_decoder_input(encoder_inputs_list, decoder_labels_list, word2index_dict, reverse_encoder_input)

encoder_inputs_t = []
decoder_inputs_t = []
decoder_labels_t = []
#tf.get_variable_scope().reuse_variables()

with tf.variable_scope('input_output'):
    for i in range(max_encoder_length):
        encoder_inputs_t.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
    for i in range(max_decoder_length):
        decoder_labels_t.append(tf.placeholder(tf.int32, shape=[None], name="decoder_label{0}".format(i)))
    decoder_inputs_t = [tf.zeros_like(decoder_labels_t[0], dtype=tf.int32, name="GO_decoder_input")] + decoder_labels_t[:-1]

with tf.variable_scope('cell'):
    single_cell = BasicLSTMCell(embeding_size)
    cell = single_cell
    if num_layers > 1:
        cell = MultiRNNCell([single_cell] * num_layers)

with tf.variable_scope('output_proj'):
    w = tf.get_variable("w", [embeding_size, len(index2word_list)])
    w_t = tf.transpose(w)
    b = tf.get_variable("b", [len(index2word_list)])
    output_projection = (w, b)

with tf.variable_scope('embedding_attention_seq2seq'):
    output, state = embedding_attention_seq2seq(
        encoder_inputs_t,
        decoder_inputs_t,
        cell,
        num_encoder_symbols=len(index2word_list),
        num_decoder_symbols=len(index2word_list),
        embedding_size=embeding_size,
        output_projection=output_projection,
        feed_previous=not(train))

##Few old code snippets
#output1 = tf.squeeze(output) # a1[0].shape: (46, 10, 300)
#output2 = tf.concat(0,output) # a2.shape: (1, 460, 300)
#output_logit1 = tf.map_fn(lambda x: tf.matmul(x,output_projection[0]) + output_projection[1], output1)
#output_logit1 = tf.matmul(output1[0], output_projection[0]) + output_projection[1] # a3
#output_logit2 = tf.matmul(output2, output_projection[0]) + output_projection[1] # a4

# as output_projection is not None, output is embedding and not the actual word
output_logit = [tf.matmul(x,output_projection[0]) + output_projection[1] for x in output]

if train:
#    
##    output = tf.squeeze(output)
#    output_logit = tf.matmul(output, output_projection[0]) + output_projection[1]
#    # tf.reshape(output,shape=(-1,-1))
##    output_logit = tf.argmax(output_logit, axis=1)
##    loss_list = []
##    for i in range(len(decoder_inputs_t)):
###        if i == len(decoder_inputs_t) - 1:
###            labels = tf.reshape(decoder_inputs_t[i], [-1, 1]) # hack for now
###        else:
###            labels = tf.reshape(decoder_inputs_t[i+1], [-1, 1])
##        labels = tf.reshape(decoder_labels_t[i], [-1, 1])
##        loss_list.append(tf.nn.sampled_softmax_loss(weights = w_t,
##                                      biases = b,
##                                      labels = labels,
##                                      inputs = output[i],
##                                      num_sampled = 10,
##                                      num_classes = len(index2word_list)))
##    loss = tf.reduce_mean(loss_list)
    loss_masking_weights = [tf.ones_like(label, dtype=tf.float32) for label in decoder_labels_t]
##     use sequence_loss for tf version less than 1.1
##     https://github.com/tensorflow/models/pull/1197
#    
    loss = sequence_loss_by_example(logits = output_logit, targets = decoder_labels_t,
                         weights = loss_masking_weights)
#
#    opt = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    opt = tf.train.AdagradOptimizer(0.001).minimize(loss)


input_feed_dict  = {}
var_init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    if train:
        sess.run(var_init)
        for step in range(epochs):
            data_index = 0;
            while(data_index + batch_size <= len(encoder_inputs_np)):
                indx = get_batch_indx(len(encoder_inputs_np), batch_size, data_index)
                input_feed_dict = get_feed_dict(encoder_inputs_np[indx], 
                                                decoder_labels_np[indx], 
                                                encoder_inputs_t, 
                                                decoder_inputs_t, 
                                                decoder_labels_t, 
                                                word2index_dict["PAD"])
                _, itr_loss = sess.run([opt,loss],input_feed_dict)
                print(itr_loss)
                data_index = data_index + batch_size;
        path=saver.save(sess, './my_test_model', global_step=1)
        print(path)
    else:
        ckpt = tf.train.get_checkpoint_state(".")
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print(ckpt.model_checkpoint_path)
            input_sent = "How are you"
            input_sent_list = input_sent.lower().strip().split(" ")
            encoded_input = [word2index_dict[x] for x in (input_sent_list + ["PAD"]* (max_encoder_length - len(input_sent_list)))]
            if reverse_encoder_input:
                encoded_input = np.flipud(encoded_input)
            for l in range(max_encoder_length):
                input_feed_dict[encoder_inputs_t[l].name] = [encoded_input[l]]
            for l in range(max_decoder_length):
                input_feed_dict[decoder_inputs_t[l].name] = [word2index_dict["GO"]]
            
#            output = tf.squeeze(tf.stack(output))
#            sentence = tf.matmul(output, output_projection[0]) + output_projection[1]
#            sentence = tf.argmax(sentence, axis=1)
#            sentence_val = sess.run(sentence,input_feed_dict)
            sentence_logits = sess.run(output_logit,input_feed_dict)
            print(" ".join([index2word_list[x[0].argsort()[-1]] for x in sentence_logits]))
#            a.argsort()[-2]
#            reply = " ".join([index2word_list[x] for x in sentence_val])
#            print(reply)
