import argparse
import random
import time
import os
import h5py
import keras
import numpy as np
import json
from keras.src.preprocessing.image import ImageDataGenerator

from constants import *

from vizualisation import plt_multiple_images

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from TensorboardCallback import TensorboardCallback


def data_generator(data_x, data_y, batch_size):
    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        shear_range=0.2,
        rotation_range=10.
    )
    datagen.fit(data_x)
    return datagen.flow(data_x, data_y, batch_size=batch_size)


def read_h5_dataset(filename, begin=0):
    with h5py.File(f'{DATASETS_DIR}/{filename}', 'r') as f:
        x_train = f['x_train'][:]
        y_train = f['y_train'][:]
        x_test = f['x_test'][:]
        y_test = f['y_test'][:]

        limit_train = begin + int(len(x_train) * SCALE)
        limit_test = begin + int(len(x_test) * SCALE)
    return (x_train[begin:limit_train], y_train[begin:limit_train],
            x_test[begin:limit_test], y_test[begin:limit_test])


def get_model(x, y, z):
    return keras.models.Sequential([
        keras.layers.Input((x, y, z)),

        keras.layers.Conv2D(96, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),

        keras.layers.Conv2D(192, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),

        keras.layers.Flatten(),
        keras.layers.Dense(1500, activation='relu'),
        keras.layers.Dropout(0.5),

        keras.layers.Dense(43, activation='softmax')]
    )


def create_callbacks(log_dir, models_dir):
    tensorboard_callback = TensorboardCallback(log_dir=f'{log_dir}/tb_{time.time()}')
    bestmodel_callback = keras.callbacks.ModelCheckpoint(filepath=models_dir + '/best_model.tf',
                                                         monitor='val_accuracy',
                                                         mode='max',
                                                         save_best_only=True)
    savemodel_callback = keras.callbacks.ModelCheckpoint(filepath=models_dir + '/{epoch:02d}.keras',
                                                         save_freq='epoch')
    return tensorboard_callback, bestmodel_callback, savemodel_callback


def callback_path():
    log_dir = f'{CALLBACK_MODELS}/{CALLBACK_LOGS}'
    models_dir = f'{CALLBACK_MODELS}/{CALLBACK_MODELS}'
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    if not os.path.isdir(models_dir):
        os.makedirs(models_dir)

    return log_dir, models_dir


def main(filename: str, multiple: bool):
    batch_size = BATCH_SIZE
    epochs = EPOCHS

    log_dir, models_dir = callback_path()
    if multiple:
        datasets = ['set-24x24-L.h5', 'set-24x24-RGB-HE.h5',
            'set-48x48-RGB.h5', 'set-24x24-L-LHE.h5',
            'set-48x48-L.h5', 'set-48x48-RGB-HE.h5',
            'set-24x24-RGB.h5', 'set-48x48-L-LHE.h5']
    else:
        datasets = [DEFAULT_DATASET]

    output = []
    for dataset in datasets:
        x_train, y_train, x_test, y_test = read_h5_dataset(dataset)
        d_size = os.path.getsize(DATASETS_DIR + '/' + dataset)/(1024*1024)
        model = get_model(x_train.shape[1], x_train.shape[2], x_train.shape[3])
        model.summary()
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # data_gen = data_generator(x_train, y_train, batch_size)
        t0 = time.time()
        tensorboard_callback, bestmodel_callback, savemodel_callback = create_callbacks(log_dir, models_dir)
        history = model.fit(x_train, y_train,
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            verbose=True,
                            callbacks=[tensorboard_callback, bestmodel_callback, savemodel_callback])
        model.save(f'{models_dir}/last-{dataset}.keras')
        t1 = time.time()
        output.append({
            'Model': dataset,
            'Data_size': d_size,
            'Accuracy': max(history.history["val_accuracy"])*100,
            'Duration': t1 - t0
        })
        print(f'Training time: {t1 - t0}')

    with open(f"{DATASETS_DIR}/{BENCHMARK_FILE}", "w") as f:
        json.dump(output, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', '-f', type=str, default=DEFAULT_DATASET)

    # --multiple will generated every models possible
    # --no-multiple use just the default dataset
    parser.add_argument('--multiple', '-m', type=str, default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    main(args.filename, args.multiple)
