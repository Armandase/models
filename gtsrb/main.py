import os
import json
import keras
import matplotlib.pyplot as plt
import numpy as np

from constants import *
from prettytable import PrettyTable
from create_model import read_h5_dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
model_path = os.path.join(MODELS_DIR, "07.keras")


def main():
    if not os.path.exists(model_path):
        print("Can't load model: ", model_path)
        exit(1)

    loaded_model = keras.models.load_model(model_path)

    with open(DATASETS_DIR + '/' + BENCHMARK_FILE, "r") as f:
        json_models = json.load(f)

        table = PrettyTable()
        table.field_names = json_models[0].keys()

        for model in json_models:
            model_name = model["Model"]
            data_size = round(model['Data_size'], 1)
            duration = round(model['Duration'], 1)
            accuracy = round(model['Accuracy'], 1)
            table.add_row([model_name, data_size, duration, accuracy])

        print(table)
    _, _, x_test, y_test = read_h5_dataset('set-48x48-RGB.h5', 3000)
    y_softmax = loaded_model.predict(x_test)
    y_pred = np.argmax(y_softmax, axis=-1)
    cm = confusion_matrix(y_pred, y_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()


if __name__ == '__main__':
    main()
