#Basic computational graph

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()  # Disable TensorFlow 2.x behavior

np.random.seed(0)

N, D = 3, 4

with tf.device('/cpu:0'):    # '/gpu:0'   Not Available Here
  x = tf.placeholder(tf.float32, shape=(N, D))
  y = tf.placeholder(tf.float32, shape=(N, D))
  z = tf.placeholder(tf.float32, shape=(N, D))

a = x + y
b = a * z
c = tf.reduce_sum(b)

grad_x, grad_y, grad_z = tf.gradients(c, [x, y, z])

with tf.Session() as sess:
    values = {
        x: np.random.rand(N, D),
        y: np.random.rand(N, D),
        z: np.random.rand(N, D),
    }
    c_val, grad_x_val, grad_y_val, grad_z_val = sess.run([c, grad_x, grad_y, grad_z], feed_dict=values)

print("c:", c_val)
print("grad_x:", grad_x_val)
print("grad_y:", grad_y_val)
print("grad_z:", grad_z_val)
