from tensorflow import keras
from tensorflow.keras import layers


# Batch Normalization
x = layers.Conv2D(32, 3, use_bias=False)(x)
x = layers.BatchNormalization()(x)
# Because the output of the Conv2D layer gets normalized, the layer doesn’t need its own bias vector

# How not to use batch normalization
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.BatchNormalization()(x)

# How to use batch normalization: the activation comes last
x = layers.Conv2D(32, 3, use_bias=False)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
# We place the activation after the BatchNormalization layer
# doing normalization before the activation maximizes the utilization of the relu.
