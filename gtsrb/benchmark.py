import json
import matplotlib.pyplot as plt
from constants import *


def main():
    benchmark_path = f"{DATASETS_DIR}/{BENCHMARK_FILE}"
    print(benchmark_path)
    with open(benchmark_path, "r") as f:
        json_models = json.load(f)
        # print(json_models)
        models_name = [model["Model"] for model in json_models]
        models_duration = [round(model["Duration"], 1) for model in json_models]
        models_accuracy = [model["Accuracy"] for model in json_models]
        plt.xlabel("Models")
        plt.xticks(range(len(models_name)), models_name)
        plt.ylabel("Duration")
        # plt.yticks(models_duration)
        plt.plot(models_duration)
        plt.show()
        # print(f.read())

if __name__ == '__main__':
    main()
