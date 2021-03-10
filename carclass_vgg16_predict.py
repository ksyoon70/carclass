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
import time

#GPU 사용시 풀어 놓을 것
"""
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
"""
IMG_SIZE = 224

THRESH_HOLD = 0.8

categories = []

result_cateories = []

dst_dirs = []

show_images = False

file_move = True


#시험 폴더 위치 지정
src_dir = './datasets/test'


base_dir = './datasets'
if not os.path.isdir(base_dir):
    os.mkdir(base_dir)
    
trained_dir = './trained'
if not os.path.isdir(trained_dir):
    os.mkdir(trained_dir)

#훈련 폴더 생성  
train_dir = os.path.join(base_dir,'train')
if not os.path.isdir(train_dir):
    os.mkdir(train_dir)
    

#훈련 폴더에서 카테고리 취득
categorie_list = os.listdir(train_dir)
categorie_list = natsort.natsorted(categorie_list)
for categorie in categorie_list:
    categories.append(categorie)

categories.append('no_categorie')

cat_len = len(categories)    
    
#결과 저장 폴더 생성    
result_dir = os.path.join(base_dir,'result')
if not os.path.isdir(result_dir):
    os.mkdir(result_dir)

#결과 폴더 아래 카테고리 디렉토리 생성
for categorie in categories:
    dst_dir = os.path.join(result_dir,categorie)
    if not os.path.isdir(dst_dir):
         os.mkdir(dst_dir)
    dst_dirs.append(dst_dir)
    
#read model
model = load_model('carclass_model.h5')
#read weight value from trained dir
weight_path = os.path.join(trained_dir,'cls_vgg16_model_weights.h5')
model.load_weights(weight_path)


total_test_files = len(os.listdir(src_dir))

print('테스트용 이미지 갯수:',total_test_files)

recog_count = 0
fail_count = 0

start_time = time.time() # strat time

if len(os.listdir(src_dir)):

    files = os.listdir(src_dir)

    for file in files:
        
        try:
            img_path = os.path.join(src_dir,file)
            img = image.load_img(img_path,target_size=(IMG_SIZE,IMG_SIZE))
            img_tensor = image.img_to_array(img)
            img_tensor = np.expand_dims(img_tensor,axis=0)
            img_tensor /= 255.
            
            preds = model.predict(img_tensor)
            
            index = np.argmax(preds[0],0)
            
            src = os.path.join(src_dir,file)
           
            
            if preds[0][index] > THRESH_HOLD :
                tilestr = 'predict: {}'.format(categories[index] ) + '' + '  probability: {:.2f}'.format(preds[0][index]*100 )  + ' %'
                dst = os.path.join(dst_dirs[index],file)
                recog_count += 1
            else:
                tilestr = 'Not sure but: {}'.format(categories[index] ) + '' + '  probability: {:.2f}'.format(preds[0][index]*100)  + ' %'
                dst = os.path.join(dst_dirs[cat_len -1 ],file)
                fail_count += 1
            #결과 디렉토리에 파일 저장
            if file_move :
                shutil.move(src,dst)
            else:
                shutil.copy(src,dst)
                
            if show_images :
                plt.title(tilestr)
                plt.imshow(img_tensor[0])
                plt.show()
        except Exception as e:
            pass
        
end_time = time.time()        
print("수행시간: {:.2f}".format(end_time - start_time))
print("건당 수행시간 : {:.2f}".format((end_time - start_time)/total_test_files))             
print('recognition: {:.2f}'.format(recog_count) +'  ({:.2f})'.format(recog_count*100/total_test_files) + ' %')       
print('fail: {}'.format(fail_count) +'  ({:.2f})'.format(fail_count*100/total_test_files) + ' %')         
        
        





