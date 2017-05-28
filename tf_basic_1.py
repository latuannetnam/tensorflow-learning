# Basic examples of tensorflow
import tensorflow as tf
import numpy as np
logs_path = '/tmp/tensorflow_logs/tf_basic_1'
# X = tf.Variable(np.array([1., 2., 3., 4.]).T, tf.float32)
input = [1, 2, 3, 4]
X = tf.placeholder(tf.float32, name='x')
W = tf.Variable([2.], tf.float32, name='W')
b = tf.Variable([3.], tf.float32, name='b')
model = tf.add(tf.multiply(X, W), b)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter(
        logs_path, graph=tf.get_default_graph())
Y = sess.run(model, {X: input})
print(np.c_[input, Y])
