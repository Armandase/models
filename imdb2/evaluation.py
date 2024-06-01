import argparse
import keras

from constants import *
from preprocessing import get_data, get_dictionary

MODELS_PATH = "/home/armand/projets/neural_networks/imdb2/models/best_model.tf"


def evaluate_with_dataset(model):
    x_train, y_train, x_test, y_test = get_data()
    x_train = keras.preprocessing.sequence.pad_sequences(x_train,
                                                         value=0,
                                                         padding='post',
                                                         maxlen=REVIEW_LEN)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test,
                                                        value=0,
                                                        padding='post',
                                                        maxlen=REVIEW_LEN)
    score = model.evaluate(x_test, y_test, verbose=0)
    print("\n\nModel evaluation:")
    print('\tx_test / loss :{:5.4f}'.format(score[0]))
    print('\tx_test / accuracy :{:5.4f}'.format(score[1]))
    score = model.evaluate(x_train, y_train, verbose=0)
    print('\tx_train / loss :{:5.4f}'.format(score[0]))
    print('\tx_train / accuracy :{:5.4f}'.format(score[1]))


def manual_evaluation(model):
    review = input('Enter a review: ')
    index_word = get_dictionary()
    word_index = {v: k for k, v in index_word.items()}
    review = review.lower()
    splitted_review = review.split()
    review = [START]
    for word in splitted_review:
        filtred_word = "".join(filter(str.isalnum, word))

        if filtred_word in word_index and word_index[filtred_word] < VOCAB_SIZE:
            review.append(word_index[filtred_word])
        else:
            review.append(UNKNOWN)
        if len(review) >= REVIEW_LEN:
            break
    print(review)
    review = keras.preprocessing.sequence.pad_sequences([review],
                                                        value=0,
                                                        padding='post',
                                                        maxlen=REVIEW_LEN)
    # print(review)
    # print(dataset2text(review, index_word))
    score = model.predict(review)
    print("\n\nModel prediction:")
    print('\treview :', *score)
    if score < 0.5:
        print('\tReview is predicted as negative')
    else:
        print('\tReview is predicted as positive')


def main(auto: bool):
    model = keras.models.load_model(MODELS_PATH)
    if auto is True:
        evaluate_with_dataset(model)
    else:
        manual_evaluation(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--auto', '-a',
                        default=True, help='Print metrics automatically')
    args = parser.parse_args()
    main(args.auto)
