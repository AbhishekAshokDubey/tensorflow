# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import re

words_around_target_word = 2;

os.chdir(r"C:\Users\Adubey4\Desktop\chatbot\code")
text_df = pd.read_csv("wiki_max_hop1.csv",index_col=[0], encoding='cp1252')
text_df.dropna(axis=0, how='any', inplace=True)

num_pattern = re.compile('[\d]+', re.UNICODE)
question_mark_pattern = re.compile(r'(\w)\1{2,}|(\?+[\w]+)|([\w]+\?+)|(\?+)', re.UNICODE)
text_df = text_df.applymap(lambda x: question_mark_pattern.sub("wrong_char" ,num_pattern.sub('num', x.lower())))

word_data = [];

for index, row in text_df.iterrows():
    print(index)
    word_tokens = [w for w in row['text'].split()]
#    word_tokens = [w for w in row['text'].split(" ") if len(w)>0]
    for i, word in enumerate(word_tokens):
        if (i>=words_around_target_word) and (i+words_around_target_word < len(word_tokens)):
            new_data_row = [word] + word_tokens[i-words_around_target_word: i] + word_tokens[i+1: i+1+words_around_target_word]
            word_data.append(new_data_row)

col_names = ["cw"] * (2*words_around_target_word+1)
for i in range(words_around_target_word):
    col_names[i+1] = "cw-"+str(words_around_target_word-i)
    col_names[2*words_around_target_word-i] = "cw+"+str(words_around_target_word-i)

word_data_df = pd.DataFrame(word_data,columns=col_names)
word_data_df.to_csv("word_dataset_cw_"+str(words_around_target_word)+".csv",index=False, encoding="UTF-8")

word_data_df.columns = [""]*len(word_data_df.columns)

word_pair_data_df = pd.concat([word_data_df.ix[:,[0, cols+1]] for cols in range(len(word_data_df.columns)-1)], axis=0)
word_pair_data_df.columns = ["cw", "cw+i"]
word_pair_data_df.to_csv("word_pair_dataset_cw_"+str(words_around_target_word)+".csv", index=False, encoding="UTF-8")
