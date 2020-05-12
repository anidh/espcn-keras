import model as espcn_model
from tensorflow.keras.models import load_model
import tensorflow_model_optimization as tfmot
import tensorflow as tf
print("Tensorflow Version",tf.__version__)

#Loading the base model
espcn = espcn_model.espcn(scale_factor=4,loader=True)
base_model = espcn()

#Loading the pretrained weights
base_model.load_weights('./models/espcn/weights_only/model_weights_010.h5')

#Helper function to quantize a model
def apply_quantization_to_conv(layer):
  if isinstance(layer, tf.keras.layers.Conv2D):
    return tfmot.quantization.keras.quantize_annotate_layer(layer)
  return layer

# Use `tf.keras.models.clone_model` to apply `apply_quantization_to_conv` 
# to the layers of the model.
annotated_model = tf.keras.models.clone_model(
    base_model,
    clone_function=apply_quantization_to_conv,
)

q_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)

# print the network structure of model
q_aware_model.summary()

#Conversion to the tflite format
converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite_model = converter.convert()
open("espcn_slim.tflite", "wb").write(quantized_tflite_model)

print("Tflite Model Exported Successfully!...")