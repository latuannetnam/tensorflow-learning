import numpy as np
import math
import random
import time
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
from UniversalFunctionApproximator import UniversalFunctionApproximator

NUM_EXAMPLES = 1000
TRAIN_SPLIT = .7

choice = 3
# train_X0 = np.array([np.linspace(-10, 10, 50)]).T
train_X0 = np.float32(
    np.random.uniform(math.pi, 6 * math.pi, (1, NUM_EXAMPLES))).T


if choice == 1:  # test set 1
    # best paramaters for function
    # single hidden layer, activation = tanh
    NUM_LAYERS = 1
    NUM_HIDDEN_NODES = 500
    MINI_BATCH_SIZE = 10
    NUM_EPOCHS = 80000
    LEARNING_RATE = 0.005
    train_Y = np.cos(train_X0) * np.sin(train_X0)
    train_X = np.c_[train_X0]

elif choice == 2:  # test set 2
    # best paramaters for function
    NUM_LAYERS = 5
    NUM_HIDDEN_NODES = 500
    MINI_BATCH_SIZE = 10
    NUM_EPOCHS = 2000
    LEARNING_RATE = 0.001
    train_Y = np.cos(train_X0) * np.sin(train_X0)
    train_X = np.c_[train_X0]


elif choice == 3:  # test set 3
    NUM_LAYERS = 5
    NUM_HIDDEN_NODES = 500
    MINI_BATCH_SIZE = 10
    NUM_EPOCHS = 50000
    # NUM_EPOCHS = 200
    LEARNING_RATE = 0.0001

    train_X1 = train_X0
    # train_X2 = train_X0 * train_X0
    train_X2 = train_X0 * 1.5
    train_Y = np.sin(train_X1 + train_X2)
    train_X = np.c_[train_X1, train_X2]

elif choice == 4:  # test set 4
    NUM_LAYERS = 2
    NUM_HIDDEN_NODES = 100
    MINI_BATCH_SIZE = 10
    NUM_EPOCHS = 2
    LEARNING_RATE = 0.001
    train_X0 = np.array([np.linspace(-10, 10, 100)]).T
    train_Y = np.cos(train_X0) * np.sin(train_X0)
    train_X = np.c_[train_X0]

    test_X = np.array([np.linspace(0, 20, 100)]).T
    train_X0 = test_X
    test_Y = np.cos(train_X0) * np.sin(train_X0)

elif choice == 5:  # test set 5
    NUM_LAYERS = 2
    NUM_HIDDEN_NODES = 100
    MINI_BATCH_SIZE = 10
    NUM_EPOCHS = 1000
    LEARNING_RATE = 0.01
    train_X0 = np.array([np.linspace(0, 10, 100)]).T
    train_Y = 0.2 + 0.4 * np.sqrt(train_X0) + 0.3 * \
        np.sin(train_X0) + 0.05 * np.cos(train_X0)
    train_X = np.c_[train_X0]

elif choice == 6:  # test set 3
    NUM_LAYERS = 5
    NUM_HIDDEN_NODES = 256
    MINI_BATCH_SIZE = 10
    # NUM_EPOCHS = 50000
    NUM_EPOCHS = 10000
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
    train_Y = np.array([blood_fat_content]).T.astype(float)
    train_X = np.array(weight_age).astype(float)

elif choice == 7:  # test set 3
    NUM_LAYERS = 2
    NUM_HIDDEN_NODES = 64
    MINI_BATCH_SIZE = 10
    # NUM_EPOCHS = 50000
    NUM_EPOCHS = 1000
    LEARNING_RATE = 0.01
    train_X1 = np.array([np.linspace(-10, 10, NUM_EXAMPLES)]).T
    train_X2 = np.array([np.linspace(-50, 50, NUM_EXAMPLES)]).T
    train_X = np.c_[train_X1, train_X2]
    train_Y = train_X1 + train_X2 * train_X1 + 3


print("X shape:", train_X.shape, " Y shape:", train_Y.shape)
ufa = UniversalFunctionApproximator(train_X, train_Y,
                                    NUM_LAYERS, NUM_HIDDEN_NODES,
                                    LEARNING_RATE,
                                    NUM_EPOCHS, MINI_BATCH_SIZE,
                                    TRAIN_SPLIT)
ufa.dump_input()
start = time.time()
ufa.fit_model()
end = time.time()
print("Duration %2.f second" % (end - start))
ufa.plot(train_X, train_Y)
