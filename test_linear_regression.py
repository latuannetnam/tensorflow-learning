import numpy as np
import math
import random
import time
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tf_linear_regression import LinearRegression

NUM_EXAMPLES = 100
TRAIN_SPLIT = .7

choice = 3
# train_X0 = np.array([np.linspace(-10, 10, 50)]).T
train_X0 = np.float32(
    np.random.uniform(math.pi, 6 * math.pi, (1, NUM_EXAMPLES))).T


if choice == 1:  # test set 1
    # best paramaters for function
    NUM_LAYERS = 1
    NUM_HIDDEN_NODES = 500
    MINI_BATCH_SIZE = 10
    NUM_EPOCHS = 2000
    LEARNING_RATE = 0.1
    train_X = np.c_[train_X0]
    train_Y = 5 * train_X0 - 10


elif choice == 2:  # test set 2
    # best paramaters for function
    NUM_LAYERS = 5
    NUM_HIDDEN_NODES = 500
    MINI_BATCH_SIZE = 10
    NUM_EPOCHS = 1000
    LEARNING_RATE = 0.001
    train_Y = np.cos(train_X0) * np.sin(train_X0)
    train_X = np.c_[train_X0]


elif choice == 3:  # test set 3
    NUM_LAYERS = 5
    NUM_HIDDEN_NODES = 500
    MINI_BATCH_SIZE = 10
    # NUM_EPOCHS = 50000
    NUM_EPOCHS = 200
    LEARNING_RATE = 0.0001

    train_X1 = train_X0
    train_X2 = train_X0 * train_X0
    train_Y = np.sin(train_X1 + train_X2)
    train_X = np.c_[train_X1, train_X2]
    test = train_X[:, 0]


elif choice == 4:  # test set 4
    NUM_LAYERS = 2
    NUM_HIDDEN_NODES = 100
    MINI_BATCH_SIZE = 10
    NUM_EPOCHS = 1000
    LEARNING_RATE = 0.001
    train_X0 = np.float32(np.array([np.linspace(-10., 10., 100)]).T)
    train_Y = np.cos(train_X0) * np.sin(train_X0)
    train_X = np.c_[train_X0]

elif choice == 5:  # test set 5
    NUM_LAYERS = 2
    NUM_HIDDEN_NODES = 100
    MINI_BATCH_SIZE = 10
    NUM_EPOCHS = 1000
    LEARNING_RATE = 0.01
    train_X0 = np.array([np.linspace(0., 10., 100)]).T
    train_Y = 0.2 + 0.4 * np.sqrt(train_X0) + 0.3 * \
        np.sin(train_X0) + 0.05 * np.cos(train_X0)
    train_X = np.c_[train_X0]

elif choice == 6:  # test set 3
    NUM_LAYERS = 5
    NUM_HIDDEN_NODES = 256
    MINI_BATCH_SIZE = 10
    # NUM_EPOCHS = 50000
    NUM_EPOCHS = 1000
    LEARNING_RATE = 0.01
    # Data from http://people.sc.fsu.edu/~jburkardt/datasets/regression/x09.txt
    weight_age = [[84, 46], [73, 20], [65, 52], [70, 30], [76, 57], [69, 25],
                  [63, 28], [72, 36], [79, 57], [75, 44], [27, 24], [89, 31],
                  [65, 52], [57, 23], [59, 60], [69, 48], [
                      60, 34], [79, 51], [75, 50],
                  [82, 34], [59, 46], [67, 23], [85, 37], [55, 40], [63, 30]]
    blood_fat_content = [354, 190, 405, 263, 451, 302, 288, 385, 402, 365, 209,
                         290, 346, 254, 395, 434, 220, 374, 308, 220, 311, 181,
                         274, 303, 244]
    train_Y = np.float32(np.array([blood_fat_content]).T)
    train_X = np.float32(np.array(weight_age))
    

print("X shape:", train_X.shape, " Y shape:", train_Y.shape)
ufa = LinearRegression(train_X, train_Y,
                       LEARNING_RATE,
                       NUM_EPOCHS, MINI_BATCH_SIZE,
                       TRAIN_SPLIT)
ufa.dump_input()
start = time.time()
ufa.fit_model()
end = time.time()
print("Duration %2.f second" % (end - start))
ufa.plot(train_X, train_Y)

