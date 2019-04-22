import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_bar(data, bins, title="", xlabel="", name=None):
    fig = plt.figure(0)
    ax = sns.barplot(x=bins[0:20], y=data, color="b")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if name:
        plt.savefig(name)
    else:
        plt.show()
    plt.close(0)


def plot_dist(data, num_bins=20, title="", xlabel="", name=None, kde=False):
    fig = plt.figure(0)
    bins = np.linspace(0, 1, num_bins+1)
    ax = sns.distplot(data, bins=bins, rug=True, kde=kde)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if name:
        plt.savefig(name)
    else:
        plt.show()
    plt.close(0)

