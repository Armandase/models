import os
import numpy as np
import argparse
# remove tf debugging mode
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from constants import *
from image_manipulation import *
from NeuralNetwork import *
from constants import *


def fashion_dataset(batch_size: int, predict=False):
    (training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

    train_dataset = tf.data.Dataset.from_tensor_slices((training_images, training_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

    train_dataset = train_dataset.map(lambda image, label: (float(image) / 255.0, label))
    test_dataset = test_dataset.map(lambda image, label: (float(image) / 255.0, label))

    if predict:
        return train_dataset, test_dataset
    train_dataset = train_dataset.batch(batch_size).shuffle(500)
    test_dataset = test_dataset.batch(batch_size).shuffle(500)
    return train_dataset, test_dataset


def train_model(train_dataset, test_dataset, learning_rate: float, epochs: int, path):
    model = NeuralNetwork()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate)
    metrics = ['accuracy']
    model.compile(optimizer, loss_fn, metrics)

    model.fit(train_dataset, epochs=epochs)

    test_loss, test_acc = model.evaluate(test_dataset)
    print(f'\nTest accuracy: {test_acc * 100:>0.1f}%, test loss: {test_loss:>8f}')

    model.save(path + '/model')


def main(batch_size: int, learning_rate: float, epochs: int, path: str):
    if not os.path.exists(path + '/model'):
        train_dataset, test_dataset = fashion_dataset(batch_size)
        train_model(train_dataset, test_dataset, learning_rate, epochs, path)
        print('Saved model to', path)
    model = tf.keras.models.load_model(path + '/model')

    train_dataset, test_dataset = fashion_dataset(batch_size, predict=True)

    pred_idx = random.randint(0, len(test_dataset) - 1)
    predict_img = list(test_dataset.as_numpy_iterator())[pred_idx]

    pred_img_array = predict_img[0]
    pred_label = predict_img[1]
    plot_array(pred_img_array, pred_label)

    img_to_pred = np.asarray(pred_img_array, dtype=np.float32).reshape((-1, 28, 28))
    predicted_vector = model.predict(img_to_pred)
    predicted_label = np.argmax(predicted_vector)
    predicted_name = LABELS_MAP[predicted_label]

    probs = tf.nn.softmax(predicted_vector.reshape((-1,)))
    for i, p in enumerate(probs):
        print(f'{LABELS_MAP[i]} -> {p * 100:.3f}%')

    print(f'Predicted class : {predicted_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fashion MNIST')
    parser.add_argument('--batch_size', '-bs', type=int, default=BATCH_SIZE)
    parser.add_argument('--learning_rate', '-lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--epochs', '-e', type=int, default=EPOCHS)
    parser.add_argument('--path', '-p', type=str, default=PATH, help='Path to save model')

    args = parser.parse_args()

    main(args.batch_size, args.learning_rate, args.epochs, args.path)
