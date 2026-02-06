import os, argparse
import matplotlib.pyplot as plt
import numpy as np
from NeuralNetwork import get_model
import keras

batch_size = 512
epochs = 16


def init_dataset():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)

    x_max = x_train.max()
    x_train = x_train / x_max
    x_test = x_test / x_max
    return x_train, y_train, x_test, y_test


def train_model(model, x_train, y_train, x_test, y_test):
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        verbose=1)

    plt.title('model accuracy')
    fig = plt.figure()
    accu_axe = fig.add_subplot(121)
    accu_axe.plot(history.history['accuracy'], label='accuracy')
    accu_axe.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend()
    loss_axe = fig.add_subplot(122)
    loss_axe.plot(history.history['loss'], label='loss')
    loss_axe.plot(history.history['val_loss'], label='val_loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.show()

    model.save('models/model.h5')


def plt_multiple_images(images, labels, labels_pred, figure_title):
    fig = plt.figure(figure_title, figsize=(10, 10))
    columns = int(np.ceil(len(images) ** 0.5))
    rows = columns + 1 if columns ** 2 <= len(images) else columns
    for i in range(1, len(images) + 1):
        img = images[i - 1]
        fig.add_subplot(rows, columns, i)
        plt.title(f'Label: {labels[i - 1]} - Prediction: {labels_pred[i - 1]}')
        plt.axis('off')
        plt.imshow(img)
    plt.show()


def main(verbose):
    x_train, y_train, x_test, y_test = init_dataset()
    model = get_model(shape=(28, 28, 1))
    model.build((1, 28, 28, 1))
    if verbose is True:
        model.summary()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    if not os.path.isdir('models/model.h5'):
        train_model(model, x_train, y_train, x_test, y_test)
    model = keras.models.load_model('models/model.h5')
    score = model.evaluate(x_test, y_test, verbose=0)

    y_softmax = model.predict(x_test)
    y_pred = np.argmax(y_softmax, axis=1)

    if verbose:
        img_verif = 10
        print(f'Test loss: {score[0]}')
        print(f'Test accuracy: {score[1]}')
        plt_multiple_images(x_test[:img_verif], y_test[:img_verif], y_pred[:img_verif], "Images predictions")

        errors = [i for i in range(len(x_test)) if y_pred[i] != y_test[i]]
        # print('Wrong index:', errors)
        failed_samples = [(x_test[i], y_test[i], y_pred[i]) for i in errors]

        plt_multiple_images(*zip(*failed_samples[:12]), "Predictions failed")


if __name__ == "__main__":
    params = argparse.ArgumentParser()
    params.add_argument(
        "--verbose", "-v", help="Option: 1=verbose mode, 0=non verbose mode(default)", default=0
    )
    args = params.parse_args()
    main(args.verbose)
