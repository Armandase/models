import os
import json
import keras
import matplotlib.pyplot as plt
import numpy as np

from constants import *
from prettytable import PrettyTable
from create_model import read_h5_dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def compare_metrics():
    best_model = (None, 0)
    with open(DATASETS_DIR + '/' + BENCHMARK_FILE, "r") as f:
        json_models = json.load(f)

        table = PrettyTable()
        table.field_names = json_models[0].keys()

        for model in json_models:
            model_name = model["Model"]
            data_size = round(model['Data_size'], 1)
            duration = round(model['Duration'], 1)
            accuracy = round(model['Accuracy'], 1)
            if best_model[1] <= accuracy:
                best_model = (model_name, accuracy)
            table.add_row([model_name, data_size, duration, accuracy])

        print(table)
    return best_model
def main():
    best_model = compare_metrics()
    model_path = os.path.join(MODELS_DIR, "best_model.tf")
    if not os.path.exists(model_path):
        print("Can't load model: ", model_path)
        exit(1)

    loaded_model = keras.models.load_model(model_path)

    _, _, x_test, y_test = read_h5_dataset(best_model[0], 3000)
    y_softmax = loaded_model.predict(x_test)
    y_pred = np.argmax(y_softmax, axis=-1)
    cm = confusion_matrix(y_pred, y_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()


if __name__ == '__main__':
    main()
