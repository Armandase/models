import matplotlib.pyplot as plt
import numpy as np


def plt_multiple_images(images, labels_pred=None, figure_title='Images'):
    fig = plt.figure(figure_title, figsize=(10, 10))
    columns = int(np.ceil(len(images) ** 0.5))
    rows = columns + 1 if columns ** 2 <= len(images) else columns
    for i in range(1, len(images) + 1):
        img = images[i - 1]
        fig.add_subplot(rows, columns, i)
        if labels_pred is not None:
            plt.title(f'{labels_pred[i - 1]}')
        plt.axis('off')
        plt.imshow(img)
    plt.show()


def images_statistics(images):
    shapes = np.array([image.shape for image in images])
    images_lx = shapes[:, 0]
    images_ly = shapes[:, 1]
    images_size = images_lx * images_ly / 1024
    images_ratio = images_lx / images_ly

    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    plt.hist(images_size, bins=100)
    plt.title('Sizes in Kpixels')
    plt.xlabel(f'Number of Kpixels')
    plt.ylabel(f'Population')
    plt.xlim([0, 30])

    plt.subplot(122)
    plt.hist(images_ratio, bins=100)
    plt.title('Ratio lx/ly for every images')
    plt.xlabel(f'Ration lx/ly')
    plt.ylabel(f'Population')
    plt.show()
