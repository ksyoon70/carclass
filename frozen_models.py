# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 22:44:31 2021

@author: user
"""
import tensorflow as tf
import os
from keras import models
from keras.models import load_model
from tensorflow.compat.v1 import graph_util
from tensorflow.python.keras import backend as K
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

trained_dir = './trained'
if not os.path.isdir(trained_dir):
    os.mkdir(trained_dir)

model = load_model('carclass_model.h5')
#read weight value from trained dir
weight_path = os.path.join(trained_dir,'cls_vgg16_model_weights.h5')
model.load_weights(weight_path)

frozen_out_path = "./frozen_models"
frozen_graph_filename = 'frozen_graph'

full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()
 
layers = [op.name for op in frozen_func.graph.get_operations()]


print("-" * 60)
print("Frozen model layers: ")
for layer in layers:
    print(layer)
print("-" * 60)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)
# Save frozen graph to disk
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=frozen_out_path,
                  name=f"{frozen_graph_filename}.pb",
                  as_text=False)
# Save its text representation
"""
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=frozen_out_path,
                  name=f"{frozen_graph_filename}.pbtxt",
                  as_text=True)
"""