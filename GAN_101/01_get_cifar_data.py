# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 10:07:54 2017

@author: ADubey4
"""

import os
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt

base_folder_path = r"C:\Users\Adubey4\Desktop\tf\GAN101"
os.chdir(base_folder_path)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

path = r"C:\Users\Adubey4\Desktop\tf\GAN101\data\cifar-10-batches-py"
data_list = os.listdir(os.path.join(base_folder_path, "data/cifar-10-batches-py"))
data_list =  [x for x in data_list if str.find(x,".")<0]

raw_image_list = []
for f in data_list:
    img_dict = unpickle(os.path.join(base_folder_path, "data/cifar-10-batches-py",f))
    raw_image_list.append(img_dict[b'data'])


all_raw_images_np = np.vstack(tuple(raw_image_list))
del raw_image_list

np.save(os.path.join(base_folder_path, "data",'cifar_all_images.npy'), all_raw_images_np)
np.savetxt(os.path.join(base_folder_path, "data",'cifar_all_images.txt'), all_raw_images_np)

#batch_images_display = all_raw_images_np.reshape(len(all_raw_images_np), 3, 32, 32).transpose(0,2,3,1).astype("uint8")
#img = Image.fromarray(batch_images_display[1], 'RGB')
#img.show()
#fig, axes1 = plt.subplots(4,4,figsize=(3,3))
#for j in range(4):
#    for k in range(4):
#        i = np.random.choice(range(len(batch_images_display)))
#        axes1[j][k].set_axis_off()
#        axes1[j][k].imshow(batch_images_display[i:i+1][0])

