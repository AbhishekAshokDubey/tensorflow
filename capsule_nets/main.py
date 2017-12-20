# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 20:58:37 2017
@author: ADubey4
tf.__version__ : '1.3.0'
"""

import os
import tensorflow as tf
#from tqdm import tqdm
from tensorflow.examples.tutorials.mnist import input_data
from capsnets import CapsNet
from data_utils import dowload_data, plot_random

import flags
import importlib
importlib.reload(flags)

print(flags.batch_size)

is_training = True

if __name__ == "__main__":
    dir_path = os.path.dirname(__file__)
    dir_path = r"C:\Users\Adubey4\Desktop\capsule"
    data_folder = os.path.join(dir_path,"data")
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print("Downloading data, this will take some time ...")
        dowload_data(data_folder)
        print("Downloading finished.")
    else:
        print("You already have the data folder. Delete it for fresh downloads")
    data = input_data.read_data_sets(data_folder, one_hot=True, reshape=False)
    # show random images
#    plot_random(data)
    model = CapsNet()
    if is_training:
        with tf.Session(graph=model.graph) as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(flags.epochs):
                x,y = data.train.next_batch(flags.batch_size)
                _, loss = sess.run([model.train_op, model.total_loss],
                                   feed_dict={model.graph.get_operation_by_name('images').outputs[0]: x, model.graph.get_operation_by_name('labels').outputs[0]: y})
                print(loss)