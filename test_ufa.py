import numpy as np
import math
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from UniversalFunctionApproximator import UniversalFunctionApproximator

NUM_HIDDEN_NODES = 100
NUM_EXAMPLES = 1000
TRAIN_SPLIT = .8
MINI_BATCH_SIZE = 100
NUM_EPOCHS = 10000
choice = 1
# train_X0 = np.array([np.linspace(-10, 10, 50)]).T
train_X0 = np.float32(
    np.random.uniform(math.pi, 6 * math.pi, (1, NUM_EXAMPLES))).T
print("X0:", min(train_X0), max(train_X0))
# standard_scaler = MinMaxScaler()
# train_X0 = standard_scaler.fit_transform(train_X0)
print("X0-scale:", min(train_X0), max(train_X0))

if choice == 1:  # test set 1
    train_Y = np.cos(train_X0) * np.sin(train_X0) - 7
    train_X = np.c_[train_X0]
    # print("X:", min(train_X0), max(train_X0))
    # print("Y", min(train_Y), max(train_Y))
elif choice == 2:  # test set 2
    train_X1 = train_X0
    train_X2 = train_X0 * train_X0
    train_Y = np.sin(train_X1 + train_X2)
    train_X = np.c_[train_X1, train_X2]


ufa = UniversalFunctionApproximator(train_X, train_Y,
                                    NUM_HIDDEN_NODES, 0.01, NUM_EPOCHS, MINI_BATCH_SIZE)
ufa.dump_input()
ufa.fit_model()
ufa.plot()
