import keras
import numpy as np

# keep 10 000 most frequently used words of the dictionnary
vocab_size = 10000
# ne retire aucun mot frequent
hide_most_frequently = 0


def get_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size,
                                                                         skip_top=hide_most_frequently)
    print(f'\nx_train: {x_train.shape}, y_train: {y_train.shape}')
    print(f'x_test: {x_test.shape}, y_test: {y_test.shape}')

    return x_train, y_train, x_test, y_test


def get_dictionary():
    index_word = keras.datasets.imdb.get_word_index()
    index_word = {w: (i + 3) for w, i in index_word.items()}
    index_word.update({'<PAD>': 0, '<START>': 1, 'UNK': 2})
    return {index: word for word, index in index_word.items()}


def dataset2text(review, index_word):
    return ' '.join([index_word.get(i, '?') for i in review])


def one_hot_encode(x, vect_size=10000):
    x_encoded = np.zeros((len(x), vect_size))

    for i, sentence in enumerate(x):
        for word in sentence:
            x_encoded[i, word] = 1
    return x_encoded
