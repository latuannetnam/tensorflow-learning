# @Author: latuannetnam@gmail.com
# class UniversalFunctionApproximator: an implementation of universal
# function approximation using TensorFlow with 1 hidden layer neural
# network
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as ax
import io
import time
import os
# from sklearn.preprocessing import StandardScaler, MinMaxScaler


class UniversalFunctionApproximator:
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    rng = np.random
    LOG_ROOT_DIR = "/tmp/tensorflow_logs/"
    NUM_THREADS = 8

    def __init__(self, input_X, label_Y,
                 n_layers=5, n_neurals=50,
                 learning_rate=0.1,
                 epochs=1000, batch_size=100,
                 split_ratio=0.8):
        self.sess = tf.Session(config=tf.ConfigProto(
            intra_op_parallelism_threads=UniversalFunctionApproximator.NUM_THREADS))
        self.logs_path = UniversalFunctionApproximator.LOG_ROOT_DIR + "ufa"
        self.input_X = input_X
        self.label_Y = label_Y
        # feature scaling to increase convergence
        # standard_scaler = MinMaxScaler()
        # print("X-pre-scale:", min(input_X), max(input_X))
        # input_X = standard_scaler.fit_transform(input_X)
        # print("X-scale:", min(input_X), max(input_X))
        # label_Y = standard_scaler.fit_transform(label_Y)

        # number of neurals per hidden layer1, if n_neurals<2: use simple linear
        # regression
        self.n_neurals = n_neurals
        self.n_layers = n_layers  # numer of hidden layers
        self.learning_rate = learning_rate  # learning rate
        self.split_ratio = split_ratio  # split ratio between training data and test data
        sample_size = input_X.shape[0]
        # size of training data
        self.train_size = int(sample_size * split_ratio)
        self.batch_size = batch_size  # batch size for training
        if batch_size >= self.train_size:
            self.batch_size = self.train_size
        #  unfix, set batch_size = train_size to increase convergence
        # self.batch_size = self.train_size

        self.epochs = epochs  # total number of training loops
        self.train_X = input_X[:self.train_size]  # train data
        self.test_X = input_X[self.train_size:]   # test data
        self.train_Y = label_Y[:self.train_size]  # train label
        self.test_Y = label_Y[self.train_size:]   # test label
        # number of features of train data: X1, X2 ...
        self.n_features = input_X.shape[1]
        with tf.variable_scope('Input', reuse=False):  # place holder for Input
            self.X = tf.placeholder("float", shape=(
                None, self.n_features), name="X")
        with tf.variable_scope('Label', reuse=False):  # place holder for Label
            self.Y = tf.placeholder("float", shape=(None), name='Y')
        if self.n_neurals < 2:  # use linear regression
            with tf.variable_scope("Weights", reuse=False):
                self.W = tf.Variable(
                    tf.zeros([self.n_features, 1]), name="weights")
                self.b = tf.Variable(tf.zeros([1]), name="bias")

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
        print("Number of hidden layers:",
              str(self.n_layers) + " neurons:" + str(self.n_neurals))
        print('Epochs:' + str(self.epochs) + " batch:" +
              str(self.batch_size) +
              " alpha:" + str(self.learning_rate))

    def inference(self, reuse=False):   # define neural network with 1 hidden layer
        if self.n_neurals < 2:  # use linear regression
            with tf.variable_scope('Model', reuse=reuse):
                model = tf.add(tf.matmul(self.X, self.W), self.b)
        else:  # use neural network
            with tf.variable_scope('Neural_Net', reuse=reuse):
                hidden1_layer = tf.layers.dense(
                    self.X, self.n_neurals, activation=tf.nn.sigmoid,
                    name="Hidden1")
                hidden2_layer = tf.layers.dense(
                    hidden1_layer, self.n_neurals, activation=tf.nn.relu6,
                    name="Hidden2")
                hidden3_layer = tf.layers.dense(
                    hidden2_layer, self.n_neurals, activation=tf.nn.relu6,
                    name="Hidden3")
                model = tf.layers.dense(
                    hidden1_layer, 1, name="Output")
        return model

    def loss(self, reuse=False):
        Y_predicted = self.inference()
        logits = tf.log(Y_predicted)
        with tf.variable_scope("Loss", reuse=reuse):
            # cost = tf.reduce_sum(tf.squared_difference(
            #     self.Y, Y_predicted))
            cost = tf.nn.l2_loss(Y_predicted - self.Y)
            # cost = tf.nn.sigmoid_cross_entropy_with_logits(
            #     None, logits=Y_predicted, labels=self.Y)
            # cost = tf.reduce_mean(cost)
        return cost

    def train(self, total_loss):
        with tf.variable_scope('Train', reuse=False):
            optimizer = tf.train.AdamOptimizer(
                self.learning_rate).minimize(total_loss)
        return optimizer

    def plot(self, show=True):
        # Graphic display
        predicted = self.sess.run(self.inference(reuse=True),
                                  feed_dict={self.X: self.input_X})
        title = 'Epochs:' + str(self.epochs) + \
            " batch:" + str(self.batch_size) + \
            " layers:" + str(self.n_layers) + \
            " neurons:" + str(self.n_neurals) + \
            " alpha:" + str(self.learning_rate)
        if self.n_features == 1:  # plot 2D image
            plt.title(title)
            # plt.plot(self.train_X, self.train_Y, c='b', label='Original data')
            # plt.plot(self.train_X, predicted, c='r', label='Fitted')
            plt.scatter(self.input_X, self.label_Y,
                        c='b', label='Original data')
            plt.scatter(self.input_X, predicted, c='r', label='Fitted')

            plt.legend()
        else:  # plot 3D for X1, X2
            if (self.n_features == 2):
                train_X1, train_X2 = np.hsplit(self.input_X, self.n_features)
            else:
                train_X1, train_X2, _ = np.hsplit(
                    self.input_X, self.n_features)
            fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            ax = fig.gca(projection='3d')
            ax.text2D(0.05, 0.95, title, transform=ax.transAxes)
            x = np.squeeze(np.asarray(train_X1))
            y = np.squeeze(np.asarray(train_X2))
            z = np.squeeze(np.asarray(self.label_Y))
            z1 = np.squeeze(np.asarray(predicted))
            # ax.plot(x, y, zs=z, c='b', label='Original data')
            # ax.plot(x, y, zs=z1, c='r', label='Fitted')
            ax.scatter(train_X1, train_X2, self.label_Y, c='b',
                       marker='s', label='Original data')
            ax.scatter(train_X1, train_X2, predicted,
                       c='r', marker='v', label='Fitted')
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
        plt.close('all')
        return buf

    def save_image(self):
        plot_buf = self.plot(show=False)
        # plot_buf = plot()
        image = tf.image.decode_png(plot_buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        # Add image summary
        image_summary_op = tf.summary.image("Function approximation", image)
        image_summary = self.sess.run(image_summary_op)
        self.summary_writer.add_summary(image_summary)

    def iterate_minibatches(self, shuffle=False):
        inputs = self.train_X
        targets = self.train_Y
        batchsize = self.batch_size
        assert inputs.shape[0] == targets.shape[0]
        if shuffle:
            indices = np.arange(inputs.shape[0])
            np.random.shuffle(indices)
            # print(indices)
        for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

    def fit_model(self):  # run loop to train model
        total_loss = self.loss()
        # Create a summary to monitor cost tensor
        tf.summary.scalar("loss", total_loss)
        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()
        train_op = self.train(total_loss), total_loss, merged_summary_op
        init = tf.global_variables_initializer()
        # Launch the graph in a session, setup boilerplate
        self.sess.run(init)
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(
            sess=self.sess, coord=self.coord)

        self.summary_writer = tf.summary.FileWriter(
            self.logs_path, graph=tf.get_default_graph())

        # actual training loop
        loss = 0
        print_step = self.epochs // 10
        for step in range(self.epochs):  # run all data
            if self.batch_size >= self.train_size:
                result = self.sess.run([train_op],
                                       feed_dict={self.X: self.train_X,
                                                  self.Y: self.train_Y})
                loss = result[0][1]
                summary = result[0][2]
                # Write logs at every iteration
                self.summary_writer.add_summary(summary, step + 1)
            else:  # run in batch_size of data
                for batch in self.iterate_minibatches(shuffle=True):
                    x_batch, y_batch = batch
                    result = self.sess.run([train_op],
                                           feed_dict={self.X: x_batch,
                                                      self.Y: y_batch})
                    loss = result[0][1]
                    summary = result[0][2]
                    # Write logs at every iteration
                    self.summary_writer.add_summary(summary, step + 1)

            if step % print_step == 0:
                loss = self.sess.run([total_loss],
                                     feed_dict={self.X: self.train_X,
                                                self.Y: self.train_Y})
                print("step:", step + 1, " loss: ", loss)

        print(" final loss:",
              self.sess.run([total_loss],
                            feed_dict={self.X: self.train_X,
                                       self.Y: self.train_Y}))

        self.save_image()
