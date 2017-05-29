# Linear regression example in TF. Multivariable linear function y = W1*x1
# + W2*x2 + b

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as ax
import io
import time
import os


class LinearRegression:
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    rng = np.random
    LOG_ROOT_DIR = "/tmp/tensorflow_logs/"
    NUM_THREADS = 8

    def __init__(self, input_X, label_Y,
                 learning_rate=0.1,
                 epochs=1000, batch_size=100,
                 split_ratio=0.8):
        # self.sess = tf.Session(config=tf.ConfigProto(
        #     intra_op_parallelism_threads=UniversalFunctionApproximator.NUM_THREADS))
        self.sess = tf.Session()
        self.logs_path = LinearRegression.LOG_ROOT_DIR + "linear_regression"

        self.input_X = input_X
        self.label_Y = label_Y

        # number of neurals per hidden layer1, if n_neurals<2: use simple linear
        # regression
        self.learning_rate = learning_rate  # learning rate
        self.split_ratio = split_ratio  # split ratio between training data and test data
        sample_size = input_X.shape[0]
        # size of training data
        self.train_size = int(sample_size * split_ratio)
        self.batch_size = batch_size  # batch size for training
        if batch_size >= self.train_size:
            self.batch_size = self.train_size
        self.epochs = epochs  # total number of training loops
        # number of features of train data: X1, X2 ...
        self.n_features = input_X.shape[1]
        self.train_X = input_X[:self.train_size]  # train data
        self.test_X = input_X[self.train_size:]   # test data
        self.train_Y = label_Y[:self.train_size]  # train label
        self.test_Y = label_Y[self.train_size:]   # test label

        with tf.variable_scope('Input', reuse=False):  # place holder for Input
            self.X = tf.placeholder("float", shape=(
                None, self.n_features), name="X")
        with tf.variable_scope('Label', reuse=False):  # place holder for Label
            self.Y = tf.placeholder("float", shape=(None), name='Y')
        with tf.variable_scope("Weights", reuse=False):
            self.W = tf.Variable(
                tf.zeros([self.n_features, 1]), dtype=tf.float32, name="weights")
            self.b = tf.Variable(tf.zeros([1]), dtype=tf.float32, name="bias")

    def __del__(self):
        self.coord.request_stop()
        self.coord.join(threads)
        self.sess.close()

    def dump_input(self):
        train = np.c_[self.train_X, self.train_Y]
        print("Train size:", train.shape[0])
        # print(train)
        test = np.c_[self.test_X, self.test_Y]
        print("test size:", test.shape[0])
        # print(test)
        print("Number of features:", self.n_features)
        print('Epochs:' + str(self.epochs) + " batch:" +
              str(self.batch_size) +
              " alpha:" + str(self.learning_rate))
        print("Number of steps:" + str(self.epochs))

    def inference(self, X, reuse=False):   # define neural network with 1 hidden layer
        with tf.variable_scope('Model', reuse=reuse):
            model = tf.add(tf.matmul(X, self.W), self.b)
        return model

    def loss(self, X, Y, reuse=False):
        Y_predicted = self.inference(X)
        logits = tf.log(Y_predicted)
        with tf.variable_scope("Loss", reuse=reuse):
            cost = tf.reduce_sum(tf.squared_difference(
                Y, Y_predicted))
            # cost = tf.nn.l2_loss(Y_predicted - self.Y)
        return cost

    def train(self, total_loss):
        with tf.variable_scope('Train', reuse=False):
            optimizer = tf.train.AdamOptimizer(
                self.learning_rate).minimize(total_loss)
        return optimizer

    def plot(self, X, Y, show=True):
        # Graphic display
        predicted = self.sess.run(self.predict_model(X))
        title = str(self.step) + ': Epochs:' + str(self.epochs) + \
            " batch:" + str(self.batch_size) + \
            " alpha:" + str(self.learning_rate)
        if self.n_features == 1:  # plot 2D image
            plt.title(title)
            # plt.plot(self.train_X, self.train_Y, c='b', label='Original data')
            # plt.plot(self.train_X, predicted, c='r', label='Fitted')
            plt.scatter(X, Y,
                        c='b', label='Original data')
            plt.scatter(X, predicted, c='r', label='Fitted')
            plt.legend()
        else:  # plot 3D for X1, X2
            if (self.n_features == 2):
                train_X1, train_X2 = np.hsplit(X, self.n_features)
            else:
                train_X1, train_X2, _ = np.hsplit(
                    X, self.n_features)
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.text2D(0.05, 0.95, title, transform=ax.transAxes)
            # line plot
            # x = np.squeeze(np.asarray(train_X1))
            # y = np.squeeze(np.asarray(train_X2))
            # z = np.squeeze(np.asarray(Y))
            # z1 = np.squeeze(np.asarray(predicted))
            # ax.plot(x, y, zs=z, c='b', label='Original data')
            # ax.plot(x, y, zs=z1, c='r', label='Fitted')

            # statter plot
            ax.scatter(train_X1, train_X2, Y, c='b',
                       marker='s', label='Original data')
            # ax.scatter(train_X1, train_X2, predicted,
            #            c='r', marker='v', label='Fitted')

            # trisurf plot
            # ax.plot_trisurf(np.ravel(train_X1),
            #                 np.ravel(train_X2), np.ravel(Y), color='b')
            ax.plot_trisurf(np.ravel(train_X1),
                            np.ravel(train_X2), np.ravel(predicted), color='r')

            ax.set_xlabel('X1')
            ax.set_ylabel('X2')
            ax.set_zlabel('Y')
            ax.legend()

        # save image to buffer
        buf = io.BytesIO()
        if show:
            # If want to view image in Tensorboard, do not show plot => Strange
            # bug!!!
            image_path = self.logs_path + "/images"
            if not os.path.exists(image_path):
                os.mkdir(image_path)
            filename = image_path + "/" + \
                time.strftime("%Y%m%d-%H%M%S", time.localtime()) + ".png"
            plt.savefig(filename, format='png')
            plt.show()
        else:
            plt.savefig(buf, format='png')
            buf.seek(0)
        plt.close()
        # plt.clf()
        return buf

    def plot_loss(self, X, Y, loss, show=True):
        # Graphic display
        title = str(self.step) + ': Epochs:' + str(self.epochs) + \
            " batch:" + str(self.batch_size) + \
            " alpha:" + str(self.learning_rate)
        train_X1 = X[:, 0]
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.text2D(0.05, 0.95, title, transform=ax.transAxes)
        # trisurf plot
        # ax.plot_trisurf(np.ravel(train_X1),
        #                 np.ravel(train_X2), np.ravel(Y), color='b')
        ax.plot_trisurf(np.ravel(train_X1),
                        np.ravel(Y), np.ravel(loss))

        ax.set_xlabel('Weight')
        ax.set_ylabel('bias')
        ax.set_zlabel('Loss')
        ax.legend()

        # save image to buffer
        buf = io.BytesIO()
        if show:
            # If want to view image in Tensorboard, do not show plot => Strange
            # bug!!!
            image_path = self.logs_path + "/images"
            if not os.path.exists(image_path):
                os.mkdir(image_path)
            filename = image_path + "/" + \
                "loss-" + \
                time.strftime("%Y%m%d-%H%M%S", time.localtime()) + ".png"
            plt.savefig(filename, format='png')
            plt.show()
        else:
            plt.savefig(buf, format='png')
            buf.seek(0)
        plt.close()
        # plt.clf()
        return buf

    def save_image(self, X, Y):
        plot_buf = self.plot(X, Y, show=False)
        # plot_buf = plot()
        image = tf.image.decode_png(plot_buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        # Add image summary
        image_summary_op = tf.summary.image(
            "Linear regression-" + str(self.step), image)
        image_summary = self.sess.run(image_summary_op)
        self.summary_writer.add_summary(image_summary)

    def predict_model(self, X):
        Y_predicted = self.inference(
            tf.convert_to_tensor(X), reuse=True)
        return Y_predicted

    def fit_model(self):  # run loop to train model
        total_loss = self.loss(self.X, self.Y)
        # Create a summary to monitor cost tensor
        tf.summary.scalar("loss", total_loss)
        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()
        train_op = self.train(total_loss), total_loss, merged_summary_op
        init = [tf.global_variables_initializer(
        ), tf.local_variables_initializer()]
        # Launch the graph in a session, setup boilerplate
        self.sess.run(init)
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(
            sess=self.sess, coord=self.coord)
        self.summary_writer = tf.summary.FileWriter(
            self.logs_path, graph=tf.get_default_graph())

        train_loss = 0
        train_all_loss = []
        train_all_W = np.array([])
        train_all_b = []
        print_step = self.epochs // 10
        if print_step == 0:
            print_step = 1
        print("Print step:" + str(print_step))
        # actual training loop
        for self.step in range(self.epochs):
            result = self.sess.run([train_op], feed_dict={
                self.X: self.train_X, self.Y: self.train_Y})
            train_loss = result[0][1]
            train_all_loss = np.append(train_all_loss, train_loss)
            summary = result[0][2]
            W, b = self.sess.run([self.W, self.b])
            if train_all_W.size == 0:
                train_all_W = W.T
            else:
                train_all_W = np.concatenate((train_all_W, W.T))
            train_all_b = np.append(train_all_b, b)
            # Write logs at every iteration
            self.summary_writer.add_summary(summary, self.step + 1)
            if self.step % print_step == 0:
                test_predicted = self.predict_model(self.test_X)
                test_loss = self.sess.run(tf.reduce_sum(tf.squared_difference(
                    self.test_Y, test_predicted)))
                self.save_image(self.train_X, self.train_Y)
                print("step:", self.step + 1, " loss: ",
                      train_loss,
                      " weight:", W.T,
                      " bias:", b,
                      " test loss:", test_loss)

        test_predicted = self.predict_model(self.test_X)
        test_loss = self.sess.run(tf.reduce_sum(tf.squared_difference(
            self.test_Y, test_predicted)))
        print("step:", self.step + 1, " loss: ",
              train_loss,
              " weight:", W.T,
              " bias:", b,
              " test loss:", test_loss)
        print(np.array([train_all_W[:, 0]]).T.shape, np.array(
            [train_all_b]).T.shape, np.array([train_all_loss]).T.shape, 
            self.train_X.shape)
        self.save_image(self.input_X, self.label_Y)
        # self.plot(self.train_X, self.train_Y)
        self.plot_loss(np.array([train_all_W[:, 0]]).T, np.array(
            [train_all_b]).T, np.array([train_all_loss]).T),
