from tensorflow import keras
import matplotlib.pyplot as plt

#  Fetching the test image
base_image_path = keras.utils.get_file(
 "coast.jpg", origin="https:/ /img-datasets.s3.amazonaws.com/coast.jpg")
plt.axis("off")
plt.imshow(keras.utils.load_img(base_image_path))

#  Instantiating a pretrained InceptionV3 model
from tensorflow.keras.applications import inception_v3
model = inception_v3.InceptionV3(weights="imagenet", include_top=False)

# Configuring the contribution of each layer to the DeepDream loss
layer_settings = {
 "mixed4": 1.0,
 "mixed5": 1.5,
 "mixed6": 2.0,
 "mixed7": 2.5,
}
outputs_dict = dict(
 [
 (layer.name, layer.output)
 for layer in [model.get_layer(name)
 for name in layer_settings.keys()]
 ]
)

feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)

# The DeepDream loss
def compute_loss(input_image):
 features = feature_extractor(input_image)
 loss = tf.zeros(shape=())
 for name in features.keys():
 coeff = layer_settings[name]
 activation = features[name]
 loss += coeff * tf.reduce_mean(tf.square(activation[:, 2:-2, 2:-2, :]))
 return loss

# The DeepDream gradient ascent process
import tensorflow as tf
@tf.function
def gradient_ascent_step(image, learning_rate):
 with tf.GradientTape() as tape:
 tape.watch(image)
 loss = compute_loss(image)
 grads = tape.gradient(loss, image)
 grads = tf.math.l2_normalize(grads)
 image += learning_rate * grads
 return loss, image
def gradient_ascent_loop(image, iterations, learning_rate, max_loss=None):
 for i in range(iterations):
 loss, image = gradient_ascent_step(image, learning_rate)
 if max_loss is not None and loss > max_loss:
 break
 print(f"... Loss value at step {i}: {loss:.2f}")
 return image

# Image processing utilities
import numpy as np
def preprocess_image(image_path):
 img = keras.utils.load_img(image_path)
 img = keras.utils.img_to_array(img)
 img = np.expand_dims(img, axis=0)
 img = keras.applications.inception_v3.preprocess_input(img)
 return img
def deprocess_image(img):
 img = img.reshape((img.shape[1], img.shape[2], 3))
 img /= 2.0
 img += 0.5
 img *= 255.
 img = np.clip(img, 0, 255).astype("uint8")
 return img

# Running gradient ascent over multiple successive "octaves"
original_img = preprocess_image(base_image_path)
original_shape = original_img.shape[1:3]
successive_shapes = [original_shape]
for i in range(1, num_octave):
     shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
     successive_shapes.append(shape)
successive_shapes = successive_shapes[::-1]
shrunk_original_img = tf.image.resize(original_img, successive_shapes[0])
img = tf.identity(original_img)
for i, shape in enumerate(successive_shapes):
     print(f"Processing octave {i} with shape {shape}")
     img = tf.image.resize(img, shape)
     img = gradient_ascent_loop(
     img, iterations=iterations, learning_rate=step, max_loss=max_loss)
     upscaled_shrunk_original_img = tf.image.resize(shrunk_original_img, shape)
     same_size_original = tf.image.resize(original_img, shape)
     lost_detail = same_size_original - upscaled_shrunk_original_img
     img += lost_detail
     shrunk_original_img = tf.image.resize(original_img, shape)
keras.utils.save_img("dream.png", deprocess_image(img.numpy()))