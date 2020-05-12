import model as espcn_model
from tensorflow.keras.models import load_model
import tensorflow_model_optimization as tfmot
import tensorflow as tf
print("TENSORFLOW",tf.__version__)
import logging
from tensorflow.compat.v1 import graph_util
from tensorflow.python.keras import backend as K
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2



#Loading the base model here
espcn = espcn_model.espcn(scale_factor=4)
base_model = espcn()
base_model.load_weights('./models/espcn/saved_weights/model_weights_001.h5')

# Helper function uses `quantize_annotate_layer` to annotate that only the 
# CONV2D layers should be quantized.
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

# Now that the Conv layers are annotated,
# `quantize_apply` actually makes the model quantization aware.
quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)
quant_aware_model.summary()
print('Input Node Name -',base_model.layers[0].name)
print('Output Node Name -',base_model.layers[-1].name)

# Save model to SavedModel format
tf.saved_model.save(quant_aware_model, "./models")

#The below code saves the pb file in tf2.0 format not comaptible with the snpe-sdk
#quant_aware_model.save('final')

full_model = tf.function(lambda x: quant_aware_model(x))
full_model = full_model.get_concrete_function(tf.TensorSpec(quant_aware_model.inputs[0].shape, quant_aware_model.inputs[0].dtype))

# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 50)
print("Frozen model layers: ")
for layer in layers:
  print(layer)

print("-" * 50)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)


# Save frozen graph from frozen ConcreteFunction to hard drive
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
              logdir="./frozen_models",
              name="espcn_slim.pb",
              as_text=False)