from preprocessing import get_data, get_dictionary
from model import create_model, get_best_model_callback
from visualize import plot_data
from constants import *
import keras


def main():
    x_train, y_train, x_test, y_test = get_data()
    x_train = keras.preprocessing.sequence.pad_sequences(x_train,
                                                         value=0,
                                                         padding='post',
                                                         maxlen=REVIEW_LEN)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test,
                                                        value=0,
                                                        padding='post',
                                                        maxlen=REVIEW_LEN)
    model = create_model()
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(x_train, y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=(x_test, y_test),
                        verbose=True,
                        callbacks=[get_best_model_callback()])
    plot_data('model accuracy',
              (history.history['accuracy'], 'accuracy'),
              (history.history['val_accuracy'], 'val_accuracy'))
    plot_data('model loss',
              (history.history['loss'], 'loss'),
              (history.history['val_loss'], 'val_loss'))

if __name__ == '__main__':
    main()
