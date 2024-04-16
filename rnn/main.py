from tensorflow import keras
from constants import *
from preprocessing import generate_dataset

def create_callback():
    save_dir = PATH + '/best_model.tf'

    savemodel_callback = keras.callbacks.ModelCheckpoint(filepath=save_dir,
                                                         monitor='val_mae',
                                                         mode='max',
                                                         save_best_only=True)
    return savemodel_callback
def main():
    train, test = generate_dataset()

    # serie to vector
    model = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=(SEQUENCE_LEN, FEATURES_LEN)),
        keras.layers.GRU(200, return_sequences=False, activation='relu'),
        keras.layers.Dense(FEATURES_LEN)
    ])
    model.summary()
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    best_model_cb = create_callback()
    history = model.fit(train,
                        epochs=EPOCHS,
                        verbose=VERBOSITY,
                        validation_data=test,
                        callbacks = [best_model_cb]
    )


if __name__ == '__main__':
    main()