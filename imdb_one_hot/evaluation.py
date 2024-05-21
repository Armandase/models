import keras
from preprocessing import get_data, one_hot_encode

MODELS_PATH = "/home/armand/projets/neural_networks/imdb1/models/best_model.tf"

def main():
    model = keras.models.load_model(MODELS_PATH)
    x_train, y_train, x_test, y_test = get_data()
    x_test = one_hot_encode(x_test)
    x_train = one_hot_encode(x_train)
    score = model.evaluate(x_test, y_test, verbose=0)
    print("\n\nModel evaluation:")
    print('\tx_test / loss :{:5.4f}'.format(score[0]))
    print('\tx_test / accuracy :{:5.4f}'.format(score[1]))
    score = model.evaluate(x_train, y_train, verbose=0)
    print('\tx_train / loss :{:5.4f}'.format(score[0]))
    print('\tx_train / accuracy :{:5.4f}'.format(score[1]))


if __name__ == '__main__':
    main()