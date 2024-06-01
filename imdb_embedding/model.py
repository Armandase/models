import keras
from constants import *


def create_model(vocab_size=VOCAB_SIZE, dense_vector_size=DENSE_VECTOR_SIZE, review_len=REVIEW_LEN):
    return keras.Sequential(
        [
            keras.layers.Input(shape=review_len),
            keras.layers.Embedding(input_dim=vocab_size,
                                   output_dim=dense_vector_size,
                                   input_length=review_len),
            keras.layers.GlobalAvgPool1D(),
            keras.layers.Dense(dense_vector_size, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ]
    )


def get_best_model_callback(path='/home/armand/projets/neural_networks/imdb2/models'):
    return keras.callbacks.ModelCheckpoint(filepath=path + '/best_model.tf',
                                           monitor='val_accuracy',
                                           mode='max',
                                           save_best_only=True)
