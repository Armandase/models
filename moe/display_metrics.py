import argparse
import numpy as np
import matplotlib.pyplot as plt 

def main(filename: list[str]) -> None:
    '''Display training metrics from log files into the same graphs.'''
    plt.figure(figsize=(12, 6))
    for fname in filename:
        epochs = []
        train_f1 = []
        test_f1 = []
        losses = []
        with open(fname, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split(',')
                epoch = int(parts[0].split()[1])
                loss = float(parts[1].split()[1])
                train_metric = float(parts[2].split()[2])
                test_metric = float(parts[3].split()[2])
                
                epochs.append(epoch)
                losses.append(loss)
                train_f1.append(train_metric)
                test_f1.append(test_metric)
        
        label_base = fname.split('_training_log.txt')[0]
        plt.subplot(1, 3, 1)
        plt.plot(epochs, train_f1, label=f'Train F1 - {label_base}')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.title('Training F1 Score Over Epochs')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(epochs, test_f1, label=f'Test F1 - {label_base}')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.title('Testing F1 Score Over Epochs')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.plot(epochs, losses, label=f'Loss - {label_base}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display training metrics from log files.")
    parser.add_argument("--filename", '-f', type=str, nargs='+', help="Path to the log file(s).")
    args = parser.parse_args()
    main(args.filename)