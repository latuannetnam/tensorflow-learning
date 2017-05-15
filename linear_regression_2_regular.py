# Linear regression with regulation example in TF.

import tensorflow as tf
import numpy as np
logs_path = '/tmp/tensorflow_logs/linear2-regulation'
W = tf.Variable(tf.zeros([2, 1]), name="weights")
b = tf.Variable(0., name="bias")
reg_lambda = 0.


def inference(X):
    return tf.matmul(X, W) + b


def loss(X, Y, W, lambda_step):
    with tf.name_scope("Model"):
        Y_predicted = inference(X)
    m = X.shape[0]
    with tf.name_scope("Loss"):
        # cost = tf.reduce_sum(tf.squared_difference(
        #     Y, Y_predicted)) / (2 * int(m))
        cost = tf.reduce_sum(tf.squared_difference(Y, Y_predicted))
        regular = tf.multiply(lambda_step, tf.reduce_sum(tf.square(W)))
        cost = tf.add(cost, regular)
        cost = tf.div(cost, 2 * int(m))
        print("lambda:", lambda_step)
    return cost


def inputs():
    # Data from http://people.sc.fsu.edu/~jburkardt/datasets/regression/x09.txt
    weight_age = [[84, 46], [73, 20], [65, 52], [70, 30], [76, 57], [69, 25],
                  [63, 28], [72, 36], [79, 57], [75, 44], [27, 24], [89, 31],
                  [65, 52], [57, 23], [59, 60], [69, 48], [
                      60, 34], [79, 51], [75, 50],
                  [82, 34], [59, 46], [67, 23], [85, 37], [55, 40], [63, 30]]
    blood_fat_content = [354, 190, 405, 263, 451, 302, 288, 385, 402, 365, 209,
                         290, 346, 254, 395, 434, 220, 374, 308, 220, 311, 181,
                         274, 303, 244]
    with tf.name_scope("Input"):
        X = tf.to_float(weight_age, name="x")
    with tf.name_scope("Label"):
        Y = tf.to_float(blood_fat_content, name='y')
    return X, Y


def train(total_loss):
    with tf.name_scope('SGD'):
        learning_rate = 0.000001
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate).minimize(total_loss)
    return optimizer


def evaluate(sess, X, Y):
    print(sess.run(inference([[80., 25.]])))  # ~ 303
    print(sess.run(inference([[65., 25.]])))  # ~ 256


def run(sess, summary_writer, training_steps, lambda_step):
    total_loss = loss(X, Y, W, reg_lambda)
    train_op = train(total_loss), merged_summary_op
    for step in range(training_steps):
        summary = sess.run([train_op])
        # print(summary[0][1])
        # Write logs at every iteration
        summary_writer.add_summary(
            summary[0][1], (step + 1) + lambda_step * training_steps)
        if step % 10 == 0:
            print("step:", (step + 1) + lambda_step * training_steps,
                  " loss: ", sess.run([total_loss]))

    print("lamda:", reg_lambda, 'final cost:', sess.run([total_loss]))


# Launch the graph in a session, setup boilerplate
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    X, Y = inputs()
    total_loss = loss(X, Y, W, reg_lambda)
    # Create a summary to monitor cost tensor
    tf.summary.scalar("loss", total_loss)
    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()
    train_op = train(total_loss), merged_summary_op
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    summary_writer = tf.summary.FileWriter(
        logs_path, graph=tf.get_default_graph())
    # actual training loop
    training_steps = 1000
    # for reg_lambda in np.linspace(0., 1., 2):
    lambda_step = 0
    for reg_lambda in (0., 0.5, 1.):
        run(sess, summary_writer, training_steps, lambda_step)
        lambda_step = lambda_step + 1
    evaluate(sess, X, Y)

    coord.request_stop()
    coord.join(threads)
    sess.close()
