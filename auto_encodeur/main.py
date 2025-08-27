import argparse
import os.path

from preprocessing import preprocessing
from model import create_model, get_best_model_callback
from constants import *
import keras
import numpy as np

import matplotlib.pyplot as plt

def plot_multiple_images(images, nb_rows=-1, labels=[]):
    fig = plt.figure('Images', figsize=(10, 10))

    if nb_rows > -1:
        rows = nb_rows
    # else:
    #     rows = columns + 1 if columns ** 2 <= len(images) else columns
    columns = int(np.ceil(len(images) / rows))
    for i in range(1, len(images) + 1):
        img = images[i - 1]
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        if not labels:
            plt.imshow(img)
        else:
            plt.imshow(img)
            plt.title(labels[i-1])
    plt.show()

def print_metrics(test_data, denoised_test, test_clean, predict_class, test_labels):
    plot_range = 5
    plot_list = list(test_data[:plot_range])
    plot_list.extend(denoised_test[:plot_range])
    plot_list.extend(test_clean[:plot_range])

    label_list = [f"input data {i}" for i in range(plot_range)]
    for i in range(plot_range):
        label_list.append(f"model output {str(predict_class[i])}")
    for i in range(plot_range):
        label_list.append(f"Real data {test_labels[i]}")
    plot_multiple_images(plot_list, 3, label_list)

def main():
    train_data, train_clean, train_labels, test_data, test_clean, test_labels = preprocessing()
    if os.path.exists(CALLBACK_DIR + 'best_model.tf'):
        model = keras.models.load_model(CALLBACK_DIR + 'best_model.tf')
        denoised_test, predict_class = model.predict(test_data)
        predict_class = np.argmax(predict_class, axis=-1)
        print_metrics(test_data, denoised_test, test_clean, predict_class, test_labels)
        return


    autoencoder_model = create_model()
    autoencoder_model.summary()
    autoencoder_model.compile(optimizer='adam',
        loss={'autoencoder': 'binary_crossentropy', 'classifier': 'sparse_categorical_crossentropy'},
        loss_weights={'autoencoder': 1., 'classifier': 1.},
        metrics={'classifier': 'accuracy'})

    history = autoencoder_model.fit(train_data, [train_clean, train_labels],
                                    epochs=EPOCHS,
                                    batch_size=BATCH_SIZE,
                                    validation_data=(test_data, [test_clean, test_labels]),
                                    callbacks=[get_best_model_callback()])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model')
    plt.show()

if __name__ == "__main__":
    main()
