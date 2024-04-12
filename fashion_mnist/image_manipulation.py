import matplotlib.pyplot as plt
import random
from constants import *


def random_subplot(cols, rows, images, labels):
    figure = plt.figure(figsize=(rows * 2, cols * 2))
    for i in range(cols * rows):
        sample_idx = random.randint(0, len(images) - 1)
        image = images[sample_idx]
        label = labels[sample_idx]

        subplot = figure.add_subplot(rows, cols, i + 1)
        subplot.set_title(LABELS_MAP[label])
        subplot.axis('off')
        subplot.imshow(image, cmap='gray')

    plt.show()


def plot_image(index, images, labels):
    image = images[index]
    label = labels[index]
    plt.title(f"idx {index}: " + LABELS_MAP[label])
    plt.axis('off')
    plt.imshow(image, cmap='gray')
    plt.show()

def plot_array(image, label):
    plt.title(LABELS_MAP[label])
    plt.axis('off')
    plt.imshow(image, cmap='gray')
    plt.show()
