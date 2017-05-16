# Logistic regression example in TF using Kaggle's Titanic Dataset.
# Download train.csv from https://www.kaggle.com/c/titanic/data

import tensorflow as tf
import numpy as np

import os
logs_path = '/tmp/tensorflow_logs/softmax-igris'

with tf.name_scope("Weights"):
    # this time weights form a matrix, not a vector, with one "feature weights
    # column" per output class.
    W = tf.Variable(tf.zeros([4, 3]), name="weights")
    # so do the biases, one per output class.
    b = tf.Variable(tf.zeros([3], name="bias"))

train_size = 100

# former inference is now used for combining inputs


def combine_inputs(X):
    combine = tf.matmul(X, W) + b
    return combine


def inference(X):
    model = tf.nn.softmax(combine_inputs(X))
    return model


def loss(X, Y):
    with tf.name_scope("Model"):
        combine = combine_inputs(X)
    with tf.name_scope("Loss"):
        cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=combine, labels=Y))
    return cost


def read_csv(batch_size, file_name, record_defaults):
    with tf.name_scope("Load-data"):
        filename_queue = tf.train.string_input_producer(
            [os.path.join(os.getcwd(), file_name)])

        reader = tf.TextLineReader(skip_header_lines=0)
        key, value = reader.read(filename_queue)

        # decode_csv will convert a Tensor from type string (the text line) in
        # a tuple of tensor columns with the specified defaults, which also
        # sets the data type for each column
        decoded = tf.decode_csv(value, record_defaults=record_defaults)

        # batch actually reads the file and loads "batch_size" rows in a single
        # tensor
        batch = tf.train.shuffle_batch(decoded,
                                       batch_size=batch_size,
                                       capacity=batch_size * 50,
                                       min_after_dequeue=batch_size,
                                       allow_smaller_final_batch=True)
    return batch


def inputs():
    sepal_length, sepal_width, petal_length, petal_width, label =\
        read_csv(train_size, "iris.data", [[0.0], [0.0], [0.0], [0.0], [""]])
    with tf.name_scope("Input"):
        # Finally we pack all the features in a single matrix;
        # We then transpose to have a matrix with one example per row and one
        # feature per column.
        features = tf.transpose(
            tf.stack([sepal_length, sepal_width, petal_length, petal_width]), name='xx')

    with tf.name_scope("Label"):
        # convert class names to a 0 based class index.
        label_number = tf.to_int32(tf.argmax(tf.to_int32(tf.stack([
            tf.equal(label, ["Iris-setosa"], name="Iris-setosa"),
            tf.equal(label, ["Iris-versicolor"], name="Iris-versicolor"),
            tf.equal(label, ["Iris-virginica"], name="Iris-virginica")
        ])), 0), name="yy")

    return features, label_number


def train(total_loss):
    with tf.name_scope('SGD'):
        learning_rate = 0.000001
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate).minimize(total_loss)
    return optimizer


def evaluate(sess, X, Y):
    with tf.name_scope("Model"):
        predicted = tf.cast(tf.arg_max(inference(X), 1), tf.int32)
    print(sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32))))


def run(sess, summary_writer, training_steps):
    for step in range(training_steps):
        summary = sess.run([train_op])
        # print(summary[0][1])
        # Write logs at every iteration
        summary_writer.add_summary(
            summary[0][1], (step + 1))
        if step % 10 == 0:
            print("step:", (step + 1),
                  " loss: ", sess.run([total_loss]))


# Launch the graph in a session, setup boilerplate
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    X, Y = inputs()
    total_loss = loss(X, Y)
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
    training_steps = 100
    run(sess, summary_writer, training_steps)
    evaluate(sess, X, Y)
    coord.request_stop()
    coord.join(threads)
    sess.close()
