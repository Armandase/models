import numpy as np
import random

from constants import *

import matplotlib.pyplot as plt
import tensorflow as tf

def visualize_data(x_train):
    random.seed()
    idx = random.randint(0, len(x_train) - SEQUENCE_LEN - PREDICT_LEN)
    print(x_train.shape)
    plt.plot(x_train[idx:idx + SEQUENCE_LEN + PREDICT_LEN, 0], label='predict sequence')
    plt.plot(x_train[idx:idx + SEQUENCE_LEN, 0], label='training sequence')
    plt.legend()
    plt.show()

def cockroach_init(s=777):
    if s > 0:
        random.seed(s)
    x_param = [ random.gauss(0,1) for i in range(8) ]
    y_param = [ random.gauss(0,1) for i in range(8) ]

    return x_param, y_param

def cockroach_move(t, x_param, y_param):
    [ax1, ax2, ax3, ax4, kx1, kx2, kx3, kx4] = x_param
    [ay1, ay2, ay3, ay4, ky1, ky2, ky3, ky4] = y_param

    x = (ax1 * np.sin(t * (kx1 + 20)) + ax2 * np.cos(t * (kx2 + 10)) +
            ax3 * np.sin(t * (kx3 + 5)) + ax4 * np.cos(t * kx4 + 5))
    y = (ay1 * np.sin(t * (ky1 + 20)) + ay2 * np.cos(t * (ky2 + 10)) +
         ay3 * np.sin(t * (ky3 + 5)) + ay4 * np.cos(t * ky4 + 5))
    return x, y

def rescale_dataset(positions):
    size = int (len(positions) * SCALE)
    dataset = np.array(positions[:size])

    size_splitted = int(len(dataset) * TRAIN_PROP)
    x_train = dataset[:size_splitted]
    x_test = dataset[size_splitted:]
    return x_train, x_test

def generate_positions():
    x_param, y_param = cockroach_init(16)
    x, y = 0, 0
    positions = []
    for t in np.arange(0, MAX_T, DELTA_T):
        x, y = cockroach_move(t, x_param, y_param)
        positions.append([x, y])
    x_train, x_test = rescale_dataset(positions)

    # Normalization
    mean = x_train.mean()
    std = x_train.std()
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    visualize_data(x_train)
    return x_train, x_test

def generate_dataset():
    x_train, x_test = generate_positions()
    train_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(x_train, x_train, length=SEQUENCE_LEN, batch_size=BATCH_SIZE)
    test_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(x_test, x_test, length=SEQUENCE_LEN, batch_size=BATCH_SIZE)
    return train_generator, test_generator