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

'''
# Deep Dream parameters
layer = 'mixed4d_3x3_bottleneck_pre_relu'
channel = 139
iterations = 20
step_size = 1.5
num_octaves = 4
octave_scale = 1.4
'''

# Function for Deep Dream
def deepdream(t_obj, img0, iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
    t_score = tf.reduce_mean(t_obj)
    t_grad = tf.gradients(t_score, t_input)[0]

    img = img0.copy()
    octaves = []

    for _ in range(octave_n - 1):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw) / octave_scale))
        hi = img - resize(lo, hw)
        img = lo
        octaves.append(hi)

    for octave in range(octave_n):
        if octave > 0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2]) + hi
        for _ in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            img += g * (step / (np.abs(g).mean() + 1e-7))

    return img

def resize(img, size):
    img = np.float32(img)
    img = PIL.Image.fromarray(np.uint8(img))
    img = img.resize((size[1], size[0]), PIL.Image.LANCZOS)  # Swap size dimensions
    return np.float32(img)


def calc_grad_tiled(img, t_grad, tile_size=512):
    sz = tile_size
    h, w = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    for y in range(0, max(h - sz // 2, sz), sz):
        for x in range(0, max(w - sz // 2, sz), sz):
            sub = img_shift[y:y+sz, x:x+sz]
            g = sess.run(t_grad, {t_input: sub})
            grad[y:y+sz, x:x+sz] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)


''' 
# Load and preprocess the input image
input_image_path = 'pilatus800.jpg'  # Replace with your input image file path
img0 = PIL.Image.open(input_image_path)
img0 = np.float32(img0)

# Apply Deep Dream
t_obj = graph.get_tensor_by_name(f'import/{layer}:0')
img_result = deepdream(t_obj[:, :, :, channel], img0, iter_n=iterations, step=step_size, octave_n=num_octaves, octave_scale=octave_scale)

# Save the output image
output_image_path = 'deepdream_output.jpg'  # Replace with your desired output image file path
img_result = np.clip(img_result, 0, 255).astype(np.uint8)
PIL.Image.fromarray(img_result).save(output_image_path)

print("Deep Dream applied and saved as", output_image_path)
'''