# Linear regression example in TF. Simple linear function y = b + W1*x1

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import io
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
rng = np.random
logs_path = '/tmp/tensorflow_logs/linear_single'
with tf.name_scope("Weights"):
    W = tf.Variable(rng.randn(), name="weights")
    b = tf.Variable(rng.randn(), name="bias")

with tf.name_scope('Input'):
    # tf Graph Input
    X = tf.placeholder("float", name='X')
with tf.name_scope('Label'):
    Y = tf.placeholder("float", name='Y')

# Training data
train_X = np.array([np.arange(100)]).T
train_Y = 5 * train_X + 1
train_size = train_X.size


def inference(X):
    return tf.add(tf.multiply(X, W), b)


def loss(X, Y):
    with tf.name_scope("Model"):
        Y_predicted = inference(X)
    with tf.name_scope("Loss"):
        cost = tf.reduce_sum(tf.squared_difference(
            Y, Y_predicted)) / (2 * train_size)

    return cost


def inputs():
    with tf.name_scope("Input"):
        # X = tf.transpose(tf.to_float(train_X), name="X")
        X = tf.to_float(train_X, name="X")
    with tf.name_scope("Label"):
        # Y = tf.transpose(tf.to_float(train_Y), name='Y')
        Y = tf.to_float(train_Y, name="Y")
    return X, Y


def train(total_loss):
    with tf.name_scope('Train'):
        # learning_rate = 0.000605
        learning_rate = 0.1
        # optimizer = tf.train.GradientDescentOptimizer(
        #     learning_rate).minimize(total_loss)
        optimizer = tf.train.AdamOptimizer(
            learning_rate).minimize(total_loss)

    return optimizer


def evaluate(sess, X, Y):
    # print(sess.run(inference([1.0])))  # 7
    # print(sess.run(inference([2.0])))  # 9
    print("--X --- Y -- predicted---")
    print(np.c_[train_X, train_Y, sess.run(
        inference(X), feed_dict={X: train_X})])


def plot():
    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    predicted = sess.run(inference(X), feed_dict={X: train_X})
    plt.plot(train_X, predicted, label='Fitted line')
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf


def save_image(summary_writer):
    plot_buf = plot()
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
tf.summary.scalar("weight", W)
tf.summary.scalar("bias", b)
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
        # for (x, y) in zip(train_X, train_Y):
        #     summary = sess.run([train_op], feed_dict={X: x, Y: y})
        summary = sess.run([train_op], feed_dict={X: train_X, Y: train_Y})
        # print(summary[0][1])
        # Write logs at every iteration
        summary_writer.add_summary(summary[0][1], step + 1)
        if step % print_step == 0:
            print("step:", step + 1, " loss: ",
                  sess.run([total_loss], feed_dict={X: train_X, Y: train_Y}),
                  " weight:", sess.run(W), " bias:", sess.run(b))

    print('W:', sess.run(W), 'b:', sess.run(b), " final loss:",
          sess.run([total_loss], feed_dict={X: train_X, Y: train_Y}))
    # evaluate(sess, X, Y)
    plot()
    save_image(summary_writer)

    coord.request_stop()
    coord.join(threads)
    sess.close()
