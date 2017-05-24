# Universal function approximator using Tensorflow NN


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as ax
import io

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
rng = np.random
logs_path = '/tmp/tensorflow_logs/ufa_nn1'

n_inputs = 1
n_hidden1 = 200
n_outputs = 1
# Training data
train_X0 = np.array([np.linspace(1, 10, 100)]).T
train_X1 = train_X0

# train_Y = 2 * train_X0 * train_X0 + np.sin(train_X0) + 7
# train_Y = 2 * np.sin(train_X0) + 7
# train_Y = 2 * train_X0 * train_X0 * train_X0 + np.sin(train_X0) + 7
train_Y = np.cos(train_X0) - 7
train_X = np.c_[train_X1]
train_size = train_X.size

with tf.name_scope('Input'):
    # tf Graph Input
    # X = tf.placeholder("float", name='X')
    X = tf.placeholder("float", shape=(None, n_inputs), name="X")
with tf.name_scope('Label'):
    Y = tf.placeholder("float", shape=(None), name='Y')


def inference(X, reuse=False):   # define neural network with 1 hidden layer
    with tf.variable_scope('Neural_Net', reuse=reuse):
        hidden1_layer1 = tf.layers.dense(
            X, n_hidden1, activation=tf.nn.tanh, name="hidden1")
        model = tf.layers.dense(hidden1_layer1, n_outputs, name="outputs")
    return model


def loss(X, Y):
    Y_predicted = inference(X)
    with tf.name_scope("Loss"):
        cost = tf.reduce_sum(tf.squared_difference(
            Y, Y_predicted)) / (2 * train_size)
        # coss = tf.nn.l2_loss(logits - Y)
    return cost


def train(total_loss):
    with tf.name_scope('Train'):
        # learning_rate = 0.000605
        learning_rate = 0.1
        optimizer = tf.train.AdamOptimizer(
            learning_rate).minimize(total_loss)
    return optimizer


def plot(show=True):
    # Graphic display
    plt.plot(train_X1, train_Y, c='b', label='Original data')
    predicted = sess.run(inference(X, reuse=True), feed_dict={X: train_X})
    plt.plot(train_X1, predicted, c='r', label='Fitted line')
    # plt.plot(train_X1, hidden, c='y', label='hidden line')
    plt.legend()
    buf = io.BytesIO()
    if show:
        # If want to view image in Tensorboard, do not show plot => Strange
        # bug!!!
        plt.show()
    else:
        # plt.savefig("/tmp/test.png", format='png')
        plt.savefig(buf, format='png')
        buf.seek(0)
    return buf


def save_image(summary_writer):
    plot_buf = plot(show=False)
    # plot_buf = plot()
    image = tf.image.decode_png(plot_buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    # Add image summary
    image_summary_op = tf.summary.image("Linear Plot", image)
    image_summary = sess.run(image_summary_op)
    summary_writer.add_summary(image_summary)

# --------- Main program --------------------


total_loss = loss(X, Y)
# Create a summary to monitor cost tensor
tf.summary.scalar("loss", total_loss)
# tf.summary.scalar("weight", W)
# tf.summary.scalar("bias", b)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()
train_op = train(total_loss), merged_summary_op
init = tf.global_variables_initializer()

# Launch the graph in a session, setup boilerplate
with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    summary_writer = tf.summary.FileWriter(
        logs_path, graph=tf.get_default_graph())
    # actual training loop
    training_steps = 1000
    print_step = training_steps // 10
    for step in range(training_steps):
        summary = sess.run([train_op], feed_dict={X: train_X, Y: train_Y})
        # print(summary[0][1])
        # Write logs at every iteration
        summary_writer.add_summary(summary[0][1], step + 1)
        if step % print_step == 0:
            print("step:", step + 1, " loss: ",
                  sess.run([total_loss], feed_dict={X: train_X, Y: train_Y}))
    print(" final loss:",
          sess.run([total_loss], feed_dict={X: train_X, Y: train_Y}))

    plot()
    save_image(summary_writer)
    coord.request_stop()
    coord.join(threads)
    sess.close()
