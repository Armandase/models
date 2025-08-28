from constants import *
import matplotlib.pyplot as plt
import csv
import argparse
import pathlib

def main(log_path=[LOG_PATH]):
    dict_data = {}

    for log_path in log_path:
        log_path = pathlib.Path(log_path)
        epochs = []
        train_losses = []
        train_accuracies = []
        with open(log_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row['epoch']))
                train_losses.append(float(row['loss']))
                train_accuracies.append(float(row['f1_score']))
            dict_data[log_path.name] = {
                'epochs': epochs,
                'train_losses': train_losses,
                'train_accuracies': train_accuracies
            }
            
    plt.figure(figsize=(12, 5)) 
    plt.subplot(1, 2, 1)
    for key, data in dict_data.items():
        plt.plot(data['epochs'], data['train_losses'], label=f'{key} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid()
    plt.subplot(1, 2, 2)
    for key, data in dict_data.items():
        plt.plot(data['epochs'], data['train_accuracies'], label=f'{key} F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('Training F1 Score over Epochs')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize training metrics")
    parser.add_argument("--log_path", type=str, nargs='+', default=[LOG_PATH], help="Path(s) to the training log file(s)")
    args = parser.parse_args()
    main(args.log_path)
