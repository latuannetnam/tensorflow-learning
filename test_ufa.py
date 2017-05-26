import numpy as np
import math
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from UniversalFunctionApproximator import UniversalFunctionApproximator

NUM_EXAMPLES = 100
TRAIN_SPLIT = .8

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
    NUM_LAYERS = 1
    NUM_HIDDEN_NODES = 500
    MINI_BATCH_SIZE = 10
    NUM_EPOCHS = 200000
    LEARNING_RATE = 0.001

    train_X1 = train_X0
    train_X2 = train_X0 * train_X0
    train_Y = np.sin(train_X1 + train_X2)
    train_X = np.c_[train_X1, train_X2]

elif choice == 3:  # test set 3
    # best paramaters for function
    NUM_LAYERS = 5
    NUM_HIDDEN_NODES = 500
    MINI_BATCH_SIZE = 20
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.005
    train_Y = np.cos(train_X0) * np.sin(train_X0)
    train_X = np.c_[train_X0]

ufa = UniversalFunctionApproximator(train_X, train_Y,
                                    NUM_LAYERS, NUM_HIDDEN_NODES,
                                    LEARNING_RATE,
                                    NUM_EPOCHS, MINI_BATCH_SIZE,
                                    TRAIN_SPLIT)
ufa.dump_input()
ufa.fit_model()
ufa.plot()
