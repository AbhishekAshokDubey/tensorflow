# -*- coding: utf-8 -*-
"""
Created on Wed May 17 08:49:21 2017

@author: ADubey4
"""
import numpy as np
import pandas as pd
import re
from collections import Counter

def get_index2word_list(df):
    all_text = " ".join(df.ix[:,0].astype(str).tolist() + df.ix[:,1].astype(str).tolist())
    all_text_list = re.split(r'\s{1,}', all_text.lower())
    return [key for key, value in Counter(all_text_list).items() if value > 1]

def get_word2index_dict(index2word_list):
    return {word: index for index, word in enumerate(index2word_list)}

def get_encoder_deconder_inputs(df, word2index_dict):
    encoder_inputs = []
    decoder_labels = []
    for index, row in df.iterrows():
        encoder_inputs.append([word2index_dict.get(x,word2index_dict["NDC"]) for x in re.split(r'\s{1,}', str(row["encoder_input"]).lower())])
        decoder_labels.append([word2index_dict.get(x,word2index_dict["NDC"]) for x in re.split(r'\s{1,}', str(row["decoder_input"]).lower())])
    return encoder_inputs, decoder_labels

def align_encoder_decoder_input(encoder_inputs_list, decoder_labels_list, word2index_dict, reverse_encoder_input=False):
    max_encoder_length = max(len(l) for l in encoder_inputs_list)
    max_decoder_length = max(len(l) for l in decoder_labels_list) + 1 # EOF
    encoder_input_padded_list=[]
    decoder_label_padded_list=[]
    for e,d in zip(encoder_inputs_list, decoder_labels_list):
        encoder_input_padded_list.append(e + [word2index_dict["PAD"]]* (max_encoder_length - len(e)))
        decoder_label_padded_list.append(d + [word2index_dict["EOF"]] + [word2index_dict["PAD"]]* (max_decoder_length - len(d)-1))
    if reverse_encoder_input:
        encoder_input = np.fliplr(np.array(encoder_input_padded_list))
    return encoder_input, np.array(decoder_label_padded_list), max_encoder_length, max_decoder_length

def get_batch_indx(data_len, batch_size = 10, start_index=None):
    if start_index==None:
        if batch_size > data_len:
            batch_size = data_len 
        indx = np.random.choice(range(data_len), size=batch_size, replace=False)
    else:
        indx = np.array(range(batch_size)) + start_index
    return indx

def get_feed_dict(encoder_inputs, label_inputs, encoder_inputs_t, decoder_inputs_t, decoder_labels_t, pad_indx = 0):
    input_feed_dict = {};
    for l in range(encoder_inputs.shape[1]):
        input_feed_dict[encoder_inputs_t[l].name] = [x[l] for x in encoder_inputs]
#    decoder_inputs  = np.zeros((label_inputs.shape[0], label_inputs.shape[1]+1))
#    decoder_inputs = np.delete(decoder_inputs,0,1)
#    if pad_indx!=0:
#        decoder_inputs[:,0] = pad_indx
    for l in range(label_inputs.shape[1]):
#        input_feed_dict[decoder_inputs_t[l].name] = [x[l] for x in decoder_inputs]
        input_feed_dict[decoder_labels_t[l].name] = [x[l] for x in label_inputs]
    return input_feed_dict