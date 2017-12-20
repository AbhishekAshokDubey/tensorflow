# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 01:39:04 2017

@author: ADubey4
"""
import wget
import os
import matplotlib.pyplot as plt
import numpy as np

# https://github.com/zalandoresearch/fashion-mnist
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py#L212
# if you are on master branch, downloading and loading can be done in one-step.
# data = input_data.read_data_sets('data/fashion',
#                                 source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')
def dowload_data(path):
    base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    for data_type in ["train", "t10k"]:
        for data_part in ["%s-labels-idx1-ubyte.gz"%data_type, "%s-images-idx3-ubyte.gz"%data_type]:
            wget.download(base_url+data_part, os.path.join(path,data_part))

def plot_random(data):
    fig = plt.figure()
    for n in range(64):
        fig.add_subplot(8,8, n + 1)
        random_image_no = np.random.choice(data.train.num_examples)
        plt.xticks([], []);plt.yticks([], [])
        plt.imshow(data.train.images[random_image_no].reshape(28,28), cmap='gray')
        plt.ylabel(str(np.argmax(data.train.labels[random_image_no])),rotation=0)
    plt.show()