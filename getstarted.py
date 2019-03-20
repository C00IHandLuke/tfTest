from __future__ import absolute_import, division, print_function

import tensorflow as tf
import time


mirrored_strategy = tf.distribute.MirroredStrategy()

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, epochs=2)

model.evaluate(x_test, y_test)

# file path to save the model
saved_model_path = "./saved_models/getstarted/{}".format(int(time.time()))

# save the model that was just created 
tf.compat.v2.keras.experimental.export_saved_model(model, saved_model_path, serving_only=True)


# # Load the saved keras model back.
new_model = tf.keras.experimental.load_from_saved_model(saved_model_path)
new_model.summary()



# load the saved keras model
model_test_new = tf.compat.v2.saved_model.load(saved_model_path)
# get the concrete function that was created when the keras model was saved
concrete_func = model_test_new.signatures[
  tf.compat.v2.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

# set the shape of the model
concrete_func.inputs[0].set_shape([1, 28, 28])

# convert to the tflite model
converter = tf.lite.TFLiteConverter.from_concrete_function(concrete_func)
tflite_model = converter.convert()

# write the model
open("converted_model.tflite", "wb").write(tflite_model)
