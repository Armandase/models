import os
import time

import h5py
import sklearn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage.filters import rank
from skimage import io, color, exposure, transform

from vizualisation import plt_multiple_images

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import tensorflow as tf

# percentage used of the dataset for the training
scale = 1
from constants import *

def read_dataset(path, shuffle=False):
    df = pd.read_csv(DEFAULT_DATASET_DIR + path)

    if 'Filename' not in df.columns or 'ClassId' not in df.columns:
        print('Wrong header/file format: need to be Filename/ClassId')
        exit(1)

    if shuffle:
        df = sklearn.utils.shuffle(df)

    img_path = os.path.dirname(DEFAULT_DATASET_DIR + path)
    images = [io.imread(f'{img_path}/{file}') for file in df['Filename']]

    return np.array(images, dtype=object), np.array(df['ClassId'])


def images_enhancement(images, width=25, height=25, proc='RGB'):
    mode = {
        'RGB': 3,
        'RGB-HE': 3,
        'L': 1,
        'L-HE': 1,
        'L-LHE': 1,
        'L-CLAHE': 1,
    }

    out = []
    for image in images:
        # check if it's a rgba color
        if image.shape[2] == 4:
            image = color.rgba2rgb(image)
        image = transform.resize(image, (width, height))

        # ---- RGB / Histogram Equalization
        if proc == 'RGB-HE':
            hsv = color.rgb2hsv(image.reshape(width, height, 3))
            hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
            image = color.hsv2rgb(hsv)

        # ---- Grayscale
        if proc == 'L':
            image = color.rgb2gray(image)

        # ---- Grayscale / Histogram Equalization
        if proc == 'L-HE':
            image = color.rgb2gray(image)
            image = exposure.equalize_hist(image)

        # ---- Grayscale / Local Histogram Equalization
        if proc == 'L-LHE':
            image = color.rgb2gray(image)
            image = img_as_ubyte(image)
            image = rank.equalize(image, disk(10)) / 255.

        # ---- Grayscale / Contrast Limited Adaptive Histogram Equalization (CLAHE)
        if proc == 'L-CLAHE':
            image = color.rgb2gray(image)
            image = exposure.equalize_adapthist(image)

        out.append(image)
    out = np.array(out, dtype='float32')
    out = out.reshape(-1, width, height, mode[proc])
    return out


def save_h5_dataset(x_train, y_train, x_test, y_test, filename):
    with h5py.File(filename, 'w') as f:
        f.create_dataset('x_train', data=x_train)
        f.create_dataset('y_train', data=y_train)
        f.create_dataset('x_test', data=x_test)
        f.create_dataset('y_test', data=y_test)
    size = os.path.getsize(filename) / (1024 * 1024)
    print(f'Dataset : {filename}  shape : {x_train.shape} size : {size} Mo   (saved)')


def cook_and_save_dataset(dirname, all_sizes=[24, 48], all_proc=['RGB', 'RGB-HE', 'L', 'L-LHE']):
    t0 = time.time()
    x_train, y_train = read_dataset("trainingset/training.csv", True)
    t1 = time.time()
    print(f'Reading data time: {t1 - t0} for training')
    x_test, y_test = read_dataset("testset/test.csv")
    t2 = time.time()
    print(f'Reading data time: {t2 - t1} for test')

    t3 = time.time()
    scaled_size_xtrain = int(len(x_train) * scale)
    scaled_size_xtest = int(len(x_test) * scale)

    for size in all_sizes:
        for proc in all_proc:
            filename = f'{dirname}/set-{size}x{size}-{proc}.h5'

            new_x_train = images_enhancement(x_train[:scaled_size_xtrain], size, size, proc)
            new_x_test = images_enhancement(x_test[:scaled_size_xtest], size, size, proc)

            save_h5_dataset(new_x_train, y_train[:scaled_size_xtrain],
                            new_x_test, y_test[:scaled_size_xtest],
                            filename)
    print(f'{time.time() - t3} taken for save {len(all_sizes) * len(all_proc)} files')


def main():
    dirname = f'{DEFAULT_DATASET_DIR}/improved_sets'
    if len(os.listdir(dirname)) == 0:
        all_sizes = [24, 48]
        all_proc = ['RGB', 'RGB-HE', 'L', 'L-LHE']
        cook_and_save_dataset(dirname, all_sizes, all_proc)
    else:
        files = os.listdir(dirname)
        diff_label = []
        diff_images = []
        for file in files:
            if not file.endswith('.h5'):
                continue
            with h5py.File(f'{dirname}/{file}', 'r') as f:
                diff_images.append(f['x_train'][0])
                diff_label.append(file)

        plt_multiple_images(diff_images, diff_label, figure_title='Images')


if __name__ == '__main__':
    main()
