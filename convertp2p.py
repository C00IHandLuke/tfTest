from __future__ import absolute_import, division, print_function

import tensorflow as tf

import os
import time
import matplotlib.pyplot as plt


mirrored_strategy = tf.distribute.MirroredStrategy()

from IPython.display import clear_output


# this file is used to convert tf models to tflite when the convert 
# does not work with gpu and needs to be run with cpu
# that means tensorflow should be set with the correct flags

#file path to model
saved_model_path = "./saved_models/p2ptest/1553040765"

#get the model
model_test_new = tf.compat.v2.saved_model.load(saved_model_path)

#get the concrete function
concrete_func = model_test_new.signatures[
  tf.compat.v2.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]


# set the shape of the function
concrete_func.inputs[0].set_shape([1,256,256,3])

# convert the model to a tflite file
converter = tf.lite.TFLiteConverter.from_concrete_function(concrete_func)
tflite_model = converter.convert()

# write the model 
open("p2pconverted_model.tflite", "wb").write(tflite_model)