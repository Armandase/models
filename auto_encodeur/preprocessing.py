import os

import matplotlib.pyplot as plt
import numpy as np

import h5py

from tensorflow import keras
from skimage.util import random_noise
from constants import *

def init_dataset():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)

    x_max = x_train.max()
    x_train = x_train / x_max
    x_test = x_test / x_max
    x_train = np.reshape(x_train, (x_train.shape[0], 28, 28, 1))
    x_test = np.reshape(x_test, (x_test.shape[0], 28, 28, 1))

    x_data = np.concatenate((x_train, x_test))
    y_data = np.concatenate((y_train, y_test))
    return x_data, y_data

def plt_multiple_images(images, figure_title):
    fig = plt.figure(figure_title, figsize=(10, 10))
    columns = int(np.ceil(len(images) ** 0.5))
    rows = columns + 1 if columns ** 2 <= len(images) else columns
    for i in range(1, len(images) + 1):
        img = images[i - 1]
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.imshow(img)
    plt.show()

def apply_noise(data):
    noised_data = data.copy()
    for i, image in enumerate(noised_data):
        image = random_noise(image, mode='gaussian', mean=0, var=0.3)
        image = random_noise(image, mode='s&p', amount=0.2, salt_vs_pepper=0.5)
        image = random_noise(image, mode='poisson')
        image = random_noise(image, mode='speckle', mean=0, var=0.1)
        noised_data[i] = image
    return noised_data

def save_hdf5_dataset(datasets, filename):
    with h5py.File(filename, 'w') as f:
        for key in datasets:
            f.create_dataset(key, data=datasets[key])
    size = os.path.getsize(filename) / (1024 * 1024)
    print(f'Dataset : {filename}, size : {size} Mo (saved)')

def reload_saved_data(file_path, shuffle=True):
    clean_data = None
    noised_data = None
    labels = None
    with h5py.File(file_path, 'r') as f:
        noised_data = np.array(f['noised_data'])
        clean_data = np.array(f['x'])
        labels = np.array(f['y'])
    if shuffle:
        np.random.seed(DATASET_SEED)
        shuffle_idx = np.random.permutation(len(noised_data))
        noised_data = noised_data[shuffle_idx]
        clean_data = clean_data[shuffle_idx]
        labels = labels[shuffle_idx]
    clean_data = clean_data[:int(len(noised_data) * SCALE)]
    noised_data = noised_data[:int(len(noised_data) * SCALE)]
    labels = labels[:int(len(labels) * SCALE)]


    train_data = noised_data[:int(len(noised_data) * TRAIN_PROPORTION)]
    test_data = noised_data[int(len(noised_data) * TRAIN_PROPORTION):]
    train_clean = clean_data[:int(len(clean_data) * TRAIN_PROPORTION)]
    test_clean = clean_data[int(len(clean_data) * TRAIN_PROPORTION):]
    train_labels = labels[:int(len(labels) * TRAIN_PROPORTION)]
    test_labels = labels[int(len(labels) * TRAIN_PROPORTION):]

    print(f'Train data: {train_data.shape}, '
          f'train prediction: {train_clean.shape}, '
          f'train labels: {train_labels.shape}')
    print(f'Test data: {test_data.shape}, '
          f'test prediction: {test_clean.shape}, '
          f'test labels: {test_labels.shape}')
    return train_data, train_clean, train_labels, test_data, test_clean, test_labels

def preprocessing():
    if not os.path.exists(DATASETS_DIR + DEFAULT_DATASET):
        x_data, y_data = init_dataset()
        print("Data collected and normalized.")
        noised_data = apply_noise(x_data)
        print("Multiple noise has been apply to the data")
        plt_multiple_images(noised_data[:10], "Noised data")
        shuffle_index = np.random.permutation(len(noised_data))
        x_data, y_data, noised_data = x_data[shuffle_index], y_data[shuffle_index], noised_data[shuffle_index]

        dict_datasets = {'x': x_data, 'y': y_data, 'noised_data': noised_data}
        save_hdf5_dataset(dict_datasets, DATASETS_DIR + DEFAULT_DATASET)

    return reload_saved_data(DATASETS_DIR + DEFAULT_DATASET)