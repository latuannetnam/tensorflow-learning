# Linear regression example in TF. Simple linear function y = b + W1*x1

import tensorflow as tf
import numpy as np
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

logs_path = '/tmp/tensorflow_logs/linear_single'
with tf.name_scope("Weights"):
    W = tf.Variable(tf.zeros([1]), name="weights")
    b = tf.Variable(0., name="bias")


def inference(X):
    return tf.multiply(W, X) + b


def loss(X, Y):
    with tf.name_scope("Model"):
        Y_predicted = inference(X)
    m = X.shape[0]
    with tf.name_scope("Loss"):
        cost = tf.reduce_sum(tf.squared_difference(
            Y, Y_predicted)) / (2 * int(m))
    return cost


def inputs():
    train_X = np.arange(100)
    train_Y = 2 * train_X + 5
    with tf.name_scope("Input"):
        X = tf.transpose(tf.to_float(train_X), name="X")
    with tf.name_scope("Label"):
        Y = tf.transpose(tf.to_float(train_Y), name='Y')
    return X, Y


def train(total_loss):
    with tf.name_scope('Train'):
        learning_rate = 0.0001
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate).minimize(total_loss)
    return optimizer


def evaluate(sess, X, Y):
    # print(sess.run(inference([1.0])))  # 7
    # print(sess.run(inference([2.0])))  # 9
    print("--X --- Y -- predicted---")
    print(np.c_[sess.run(X), sess.run(Y), sess.run(inference(X))])


# Launch the graph in a session, setup boilerplate
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    X, Y = inputs()
    # print(np.c_[sess.run(X), sess.run(Y)])

    total_loss = loss(X, Y)

    # Create a summary to monitor cost tensor
    tf.summary.scalar("loss", total_loss)
    # tf.summary.scalar("weights", W[0])
    # tf.summary.scalar("bias", b[0])
    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    train_op = train(total_loss), merged_summary_op

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    summary_writer = tf.summary.FileWriter(
        logs_path, graph=tf.get_default_graph())
    # actual training loop
    training_steps = 2000
    for step in range(training_steps):
        summary = sess.run([train_op])
        # print(summary[0][1])
        # Write logs at every iteration
        summary_writer.add_summary(summary[0][1], step + 1)
        if step % 100 == 0:
            print("step:", step + 1, " loss: ",
                  sess.run([total_loss]), " weight:", sess.run(W), " bias:", sess.run(b))
    print('W:', sess.run(W), 'b:', sess.run(b))
    evaluate(sess, X, Y)

    coord.request_stop()
    coord.join(threads)
    sess.close()
