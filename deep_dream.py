# backend of deep dream application
# import's google's inception model to recreate deep dream prog.

import numpy as np
from functools import partial
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt
import urllib.request
import os
import zipfile

# Download Google's Inception5h model
url = 'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'
filename = 'inception5h.zip'
model_dir = 'inception5h'
if not os.path.exists(model_dir):
    urllib.request.urlretrieve(url, filename)
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(model_dir)
    os.remove(filename)

# Load the Inception5h model
model_fn = os.path.join(model_dir, 'tensorflow_inception_graph.pb')
graph = tf.compat.v1.Graph()
sess = tf.compat.v1.InteractiveSession(graph=graph)
with tf.io.gfile.GFile(model_fn, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
t_input = tf.compat.v1.placeholder(np.float32, name='input')
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)
tf.compat.v1.import_graph_def(graph_def, {'input': t_preprocessed})

# Apply the Deep Dream algorithm to modify an input image based on a target tensor.
# Args:
#   t_obj: TensorFlow tensor representing the target activation to maximize.
#   img0: Input image as a NumPy array.
#   iter_n: Number of iterations for each octave (default: 10).
#   step: Step size for gradient ascent (default: 1.5).
#   octave_n: Number of octaves to generate (default: 4).
#   octave_scale: Scaling factor for octaves (default: 1.4).
# Returns:
#   Modified image as a NumPy array.
def deepdream(t_obj, img0, iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
    
    # Calculate the score and gradient with respect to the input tensor
    t_score = tf.reduce_mean(t_obj)
    t_grad = tf.gradients(t_score, t_input)[0]

    # Create a copy of the input image
    img = img0.copy()
    octaves = []

    # Generate multiple octaves of the image by resizing and calculating differences
    for _ in range(octave_n - 1):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw) / octave_scale))
        hi = img - resize(lo, hw)
        img = lo
        octaves.append(hi)

    # Apply Deep Dream algorithm on each octave of the image
    for octave in range(octave_n):
        if octave > 0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2]) + hi
        for _ in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            img += g * (step / (np.abs(g).mean() + 1e-7))

    return img
# end 

# Resize the input image using Lanczos interpolation to the specified size.
# Args:
#   img: Input image as a NumPy array.
#   size: Target size (height, width) for resizing.
# Returns:
#   Resized image as a float32 NumPy array.
def resize(img, size):
    # Convert image to float32 and create a PIL.Image object
    img = np.float32(img)
    img = PIL.Image.fromarray(np.uint8(img))

    # Resize the image using Lanczos interpolation
    img = img.resize((size[1], size[0]), PIL.Image.LANCZOS)  # Swap size dimensions
    return np.float32(img)

# Calculate the gradient of an image with respect to a given tensor using a tiled approach.
# Args:
#   img: Input image as a NumPy array.
#   t_grad: TensorFlow tensor representing the gradient.
#   tile_size: Size of the tiles for calculating gradients (default: 512).
# Returns:
#   Combined gradient of the image as a NumPy array.
def calc_grad_tiled(img, t_grad, tile_size=512):
    sz = tile_size
    h, w = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    
    # Calculate gradients for image tiles and combine them
    for y in range(0, max(h - sz // 2, sz), sz):
        for x in range(0, max(w - sz // 2, sz), sz):
            sub = img_shift[y:y+sz, x:x+sz]
            g = sess.run(t_grad, {t_input: sub})
            grad[y:y+sz, x:x+sz] = g

    # Roll the gradients to align with the original image
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)

