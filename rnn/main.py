import os
import random

from tensorflow import keras
from constants import *
from preprocessing import generate_dataset
import matplotlib.pyplot as plt
import numpy as np

def create_callback():
    save_dir = PATH + '/best_model.tf'

    savemodel_callback = keras.callbacks.ModelCheckpoint(filepath=save_dir,
                                                         monitor='val_mae',
                                                         mode='max',
                                                         save_best_only=True)
    return savemodel_callback

def create_model():
    model = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=(SEQUENCE_LEN, FEATURES_LEN)),
        # keras.layers.LSTM(200, return_sequences=False, activation='relu'),
        keras.layers.GRU(200, return_sequences=False, activation='relu'),
        keras.layers.Dense(FEATURES_LEN)
    ])
    if VERBOSITY:
        model.summary()
    return model

def train_model(model, train, test):
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    best_model_cb = create_callback()
    history = model.fit(train,
                        epochs=EPOCHS,
                        verbose=VERBOSITY,
                        validation_data=test,
                        callbacks=[best_model_cb]
    )
    return history

def prediction(model, test):
    s = random.randint(0, len(test) - SEQUENCE_LEN * 2)
    predict_list_x = []
    predict_list_y = []

    model_seq = list(test[s: s + SEQUENCE_LEN])
    predict_list_model_seq = []
    for i in range(SEQUENCE_LEN):
        sequence = test[s + i :s + i + SEQUENCE_LEN]
        predicted_values = model.predict(np.array([sequence]), verbose=VERBOSITY)
        predict_model_seq = model.predict(np.array([model_seq]), verbose=VERBOSITY)
        predict_list_model_seq.append(predict_model_seq[0])
        predict_list_x.append(predicted_values[0][0])
        predict_list_y.append(predicted_values[0][1])
        model_seq.append(predicted_values[0])

    x_pred = [pred[0] for pred in predict_list_model_seq]
    y_pred = [pred[1] for pred in predict_list_model_seq]
    plt.plot(x_pred, y_pred, label='model sequence')
    plt.plot(predict_list_x, predict_list_y, label='prediction')
    true_list = test[s + SEQUENCE_LEN: s + SEQUENCE_LEN * 2]
    plt.plot(true_list[:,0], true_list[:,1], label='true')
    plt.legend()
    plt.show()
def main(): 
    train, test, x_train, x_test = generate_dataset()
    if os.path.exists(PATH + '/best_model.tf'):
        model = keras.models.load_model(PATH + '/best_model.tf')
    else:
        model = create_model()
        history = train_model(model, train, test)
        if VERBOSITY > 0:
            plt.plot(history.history['loss'], label='mse')
            plt.plot(history.history['val_loss'], label='val_mse')
            plt.legend()
            plt.show()

    prediction(model, x_test)

if __name__ == '__main__':
    main()