import tensorflow as tf
from keras.models import load_model

input_file = "model.h5"
output_file = "model.tflite"

model = load_model(input_file)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
open(output_file, "wb").write(tflite_model)