# Linear regression with simple neural network
# y= sum(X*W + b)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as ax
import io

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
rng = np.random
logs_path = '/tmp/tensorflow_logs/linear_neuralnet'

n_inputs = 1
n_hidden1 = 2
n_outputs = 1
# Training data
train_X0 = np.array([np.linspace(1, 10, 100)]).T
train_X1 = train_X0
train_X2 = train_X0 * train_X0
# train_Y = 2 * train_X0 * train_X0 + 7
train_Y = 2 * np.sin(train_X0) + 7

train_X = np.c_[train_X1]
# train_X = np.c_[train_X1, train_X1]
train_size = train_X.size

with tf.name_scope('Input'):
    # tf Graph Input
    # X = tf.placeholder("float", name='X')
    X = tf.placeholder("float", shape=(None, n_inputs), name="X")
with tf.name_scope('Label'):
    Y = tf.placeholder("float", shape=(None), name='Y')


def plot(show=True):
    # Graphic display
    plt.plot(train_X1, train_Y, c='b', label='Original data')
    hidden = hidden1.eval(feed_dict={X: train_X})
    predicted = logits.eval(feed_dict={X: train_X})
    plt.plot(train_X1, predicted, c='r', label='Fitted line')
    plt.plot(train_X1, hidden, c='y', label='hidden line')
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


def plot3D(show=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax = fig.gca(projection='3d')
    predicted = logits.eval(feed_dict={X: train_X})
    x = np.squeeze(np.asarray(train_X1))
    y = np.squeeze(np.asarray(train_X2))
    z = np.squeeze(np.asarray(train_Y))
    z1 = np.squeeze(np.asarray(predicted))
    ax.plot(x, y, zs=z, c='b', label='Original')
    ax.plot(x, y, zs=z1, c='r', label='Fit')
    # ax.scatter(train_X1, train_X2, train_Y, c='y', marker='s')
    # ax.scatter(train_X1, train_X2, predicted, c='r', marker='v')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    ax.legend()
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


# with tf.name_scope("dnn"):
#     hidden1 = neuron_layer(X, n_hidden1, "hidden1")
#     logits = neuron_layer(hidden1, n_outputs, "outputs")
with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
    logits = tf.layers.dense(hidden1, n_outputs, name="outputs")

with tf.name_scope("loss"):
    loss = tf.reduce_sum(tf.squared_difference(
            Y, logits)) / (2 * train_size)

learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

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
        # summary_writer.add_summary(summary[0][1], step + 1)
        if step % print_step == 0:
            print("step:", step + 1, " loss: ",
                  sess.run([loss], feed_dict={X: train_X, Y: train_Y}))
            # print("step:", step + 1, " loss: ",
            #       sess.run([loss], feed_dict={X: train_X, Y: train_Y}),
            #       " weight:", sess.run(W).T, " bias:", sess.run(b))
            # plot3D()

    print(" final loss:",
          sess.run([loss], feed_dict={X: train_X, Y: train_Y}))

    # plot3D()
    plot()

    # print('W:', sess.run(W).T, 'b:', sess.run(b), " final loss:",
    #       sess.run([total_loss], feed_dict={X: train_X, Y: train_Y}))
