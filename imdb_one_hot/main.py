import matplotlib.pyplot as plt

from preprocessing import get_data, get_dictionary, dataset2text, one_hot_encode
from model import create_model, get_best_model_callback
epochs = 10
batch_size = 512
vocab_size = 10000
exemple_idx = 0

def main():
    x_train, y_train, x_test, y_test = get_data()
    index_word = get_dictionary()
    print("Sentence exemple: ", dataset2text(x_train[exemple_idx], index_word))
    x_train = one_hot_encode(x_train)
    x_test = one_hot_encode(x_test)
    print("Encoded exemple: ", x_train[exemple_idx])
    model = create_model(vocab_size)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_test, y_test),
                        verbose=True,
                        callbacks=[get_best_model_callback()])
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.legend()
    plt.show()
if __name__ == '__main__':
    main()
