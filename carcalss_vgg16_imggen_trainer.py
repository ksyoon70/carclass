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
import random
from keras.applications import VGG16
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from datetime import datetime
from sklearn.utils import class_weight
import natsort


#GPU 사용시 풀어 놓을 것
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

original_dataset_dir = './datasets/training_set'
#categories = ["class1","class2","class3","class45","class6"]
categories = []
y_train = []
backbone = "vgg16"

#이미지 크기 조정 크기
IMG_SIZE = 224
#배치 싸이즈
BATCH_SIZE = 20
#epochs
EPOCHS =  50

# train data count
train_data_count = 0

# validation data count
val_data_count = 0


def get_model_path(model_type, backbone="vgg16"):
    """Generating model path from model_type value for save/load model weights.
    inputs:
        model_type = "rpn", "faster_rcnn"
        backbone = "vgg16", "mobilenet_v2"
    outputs:
        model_path = os model path, for example: "trained/rpn_vgg16_model_weights.h5"
    """
    main_path = "trained"
    if not os.path.exists(main_path):
        os.makedirs(main_path)
    model_path = os.path.join(main_path, "{}_{}_model_weights.h5".format(model_type, backbone))
    return model_path

def get_log_path(model_type, backbone="vgg16", custom_postfix=""):
    """Generating log path from model_type value for tensorboard.
    inputs:
        model_type = "rpn", "faster_rcnn"
        backbone = "vgg16", "mobilenet_v2"
        custom_postfix = any custom string for log folder name
    outputs:
        log_path = tensorboard log path, for example: "logs/rpn_mobilenet_v2/{date}"
    """
    return "logs/{}_{}{}/{}".format(model_type, backbone, custom_postfix, datetime.now().strftime("%Y%m%d-%H%M%S"))
#------------- 영상 생성 및 디렉토리 시작 --------------

base_dir = './datasets'
if not os.path.isdir(base_dir):
    os.mkdir(base_dir)

train_dir = os.path.join(base_dir,'train')
if not os.path.isdir(train_dir):
    os.mkdir(train_dir)

validation_dir = os.path.join(base_dir,'validation')
if not os.path.isdir(validation_dir):
    os.mkdir(validation_dir)

test_dir = os.path.join(base_dir,'test')
if not os.path.isdir(test_dir):
    os.mkdir(test_dir)

categorie_list = os.listdir(train_dir)
categorie_list = natsort.natsorted(categorie_list)
for categorie in categorie_list:
    categories.append(categorie)

index = 0    
for categorie in categories:
    catpath = os.path.join(train_dir,categorie)
    valpath = os.path.join(validation_dir,categorie)
    #y_train += [index] * len(os.listdir(catpath))
    #get training data file count
    train_data_count += len(os.listdir(catpath))
    val_data_count += len(os.listdir(valpath))
    index = index + 1
    
    
    
train_datagen = ImageDataGenerator(
                            rescale=1./255,
                            rotation_range=5,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2)

#train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)




train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(IMG_SIZE,IMG_SIZE),
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    seed=42,
                                                    class_mode='categorical',
                                                    classes = categories)


validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                    target_size=(IMG_SIZE,IMG_SIZE),
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    seed=42,
                                                    class_mode='categorical',
                                                    classes = categories)


print(train_generator.class_indices)
  
conv_base = VGG16(weights='imagenet',
                  include_top = False,
                  input_shape=(IMG_SIZE,IMG_SIZE,3))
#conv_base.summary()


# Convolution Layer를 학습되지 않도록 고정 
for layer in conv_base.layers:
    layer.trainable = False 


model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(len(categories),activation='softmax'))

#model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])
model.compile(loss="categorical_crossentropy",
                              optimizer="adam",metrics=["acc"])
#model.summary()

# Load weights
log_path = get_log_path("cls", backbone)
model_path = get_model_path("cls")
checkpoint_callback = ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, save_weights_only=True)
tensorboard_callback = TensorBoard(log_dir=log_path)

"""
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
"""
class_weights = class_weight.compute_class_weight(
               'balanced',
                np.unique(train_generator.classes), 
                train_generator.classes) 

class_weights = {i : class_weights[i] for i in range(len(categories))}

#calculate steps_per_epoch and validation_steps
steps_per_epoch = int(train_data_count/BATCH_SIZE)
validation_steps = int(val_data_count/BATCH_SIZE)

history = model.fit(train_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=EPOCHS,
                              validation_data=validation_generator,
                              validation_steps=validation_steps,
                              class_weight=class_weights,
                              callbacks=[checkpoint_callback, tensorboard_callback])


model.save('carclass_model.h5',)



acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)


plt.plot(epochs, acc, 'bo', label ='Training acc')
plt.plot(epochs, val_acc, 'b', label ='Validation acc')

plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label ='Training loss')
plt.plot(epochs, val_loss, 'b', label ='Validation loss')

plt.title('Training and validation loss')
plt.legend()
plt.figure()

plt.show()

