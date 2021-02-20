#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 20:22:21 2021

@author: headway
"""
import numpy as np
import os, shutil
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.applications import VGG16
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import load_model
import natsort

#GPU 사용시 풀어 놓을 것
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

IMG_SIZE = 224

categories = []

test_dir = './datasets/test'


base_dir = './datasets'
if not os.path.isdir(base_dir):
    os.mkdir(base_dir)

train_dir = os.path.join(base_dir,'train')
if not os.path.isdir(train_dir):
    os.mkdir(train_dir)


categorie_list = os.listdir(train_dir)
categorie_list = natsort.natsorted(categorie_list)
for categorie in categorie_list:
    categories.append(categorie)

model = load_model('carclass_1.h5')

print('테스트용 이미지 갯수:',len(os.listdir(test_dir)))

if len(os.listdir(test_dir)):

    files = os.listdir(test_dir)
    for file in files:
        
        try:
            img_path = os.path.join(test_dir,file)
            img = image.load_img(img_path,target_size=(IMG_SIZE,IMG_SIZE))
            img_tensor = image.img_to_array(img)
            img_tensor = np.expand_dims(img_tensor,axis=0)
            img_tensor /= 255.
            
            preds = model.predict(img_tensor)
            
            index = np.argmax(preds[0],0)
            
            tilestr = 'predict: {}'.format(categories[index] ) + '' + '  probability: {}'.format(str(preds[0][index]*100) )  + ' %'

        
            plt.title(tilestr)
            plt.imshow(img_tensor[0])
            plt.show()
        except Exception as e:
            pass
                
        
        
        
        





