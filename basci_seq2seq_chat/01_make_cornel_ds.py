# -*- coding: utf-8 -*-
"""
Created on Sun May 14 17:27:41 2017

@author: ADubey4
"""

import os
import pandas as pd
import re
import ast
import itertools

def process_line(line):
    return re.sub("[-.,?'!/<>=;_\"]", " ", str(line)).strip();

relative_path_to_cornell_ds_list = ['data', "cornell movie-dialogs corpus"]

base_folder_path = os.path.dirname(__file__)
os.chdir(base_folder_path)

movie_lines_file_path_list = [base_folder_path] + relative_path_to_cornell_ds_list + ["movie_lines.txt"]
movie_conversations_file_path_list = [base_folder_path] + relative_path_to_cornell_ds_list + ["movie_conversations.txt"]

movie_lines_file_path = os.path.join(*movie_lines_file_path_list)
movie_lines_df = pd.read_csv(movie_lines_file_path, sep=re.escape('+++$+++'), engine='python', header=None)
movie_lines_df[0] = movie_lines_df[0].str.strip()
movie_lines_df.set_index(0, inplace=True, drop=True)

movie_conversations_file_path = os.path.join(*movie_conversations_file_path_list)
movie_conversations_df = pd.read_csv(movie_conversations_file_path, sep=re.escape('+++$+++'), engine='python', header=None)

line_seq_str_list_series = movie_conversations_df.ix[:,movie_conversations_df.columns[-1]]
line_seq_list_series = (line_seq_str_list_series.apply(lambda x: ast.literal_eval(x.strip())))
line_seq_list = list(itertools.chain.from_iterable(line_seq_list_series))
#line_seq_list = [item for sublist in line_seq_list_series for item in sublist]

line_seq_list_itr = iter(line_seq_list)
line_seq_dialog_list = zip(line_seq_list_itr,line_seq_list_itr)

dialog_list = [[process_line(movie_lines_df.loc[d1].iloc[-1]),process_line(movie_lines_df.loc[d2].iloc[-1])] for d1, d2 in line_seq_dialog_list]

dialog_df = pd.DataFrame(dialog_list, columns=["encoder_input", "decoder_input"])
dialog_df.to_csv("cornell_raw.csv",index=None, encoding='utf-8')