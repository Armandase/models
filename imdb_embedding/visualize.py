import matplotlib.pyplot as plt


def plot_data(title, *data):
    for plotted, label in data:
        plt.plot(plotted, label=label)
    plt.title(title)
    plt.legend()
    plt.show()
