import keras
import numpy as np

from constants import *


def get_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=VOCAB_SIZE,
                                                                         skip_top=HIDE_MOST_FREQUENTLY,
                                                                         seed=42)
    print('Max(x_train, x_test) : ', np.max(np.max([x_train, x_test])))
    print('Min(x_train, x_test) : ', np.min(np.min([x_train, x_test])))
    print(f'x_train: {x_train.shape}, y_train: {y_train.shape}')
    print(f'x_test: {x_test.shape}, y_test: {y_test.shape}')

    y_train = np.asarray(y_train).astype('float32')
    y_test = np.asarray(y_test).astype('float32')
    return x_train, y_train, x_test, y_test


def get_dictionary():
    index_word = keras.datasets.imdb.get_word_index()
    index_word = {w: (i + 3) for w, i in index_word.items()}
    index_word.update({'<PAD>': 0, '<START>': 1, '<UNKNOWN>': 2, '<UNDEF>': 3})
    return {index: word for word, index in index_word.items()}


def dataset2text(review, index_word):
    return ' '.join([index_word.get(i, '?') for i in review])


def one_hot_encode(x, vect_size=10000):
    x_encoded = np.zeros((len(x), vect_size))

    for i, sentence in enumerate(x):
        for word in sentence:
            x_encoded[i, word] = 1
    return x_encoded
