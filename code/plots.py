from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_plot_dir(output_dir):
    plot_path = Path(output_dir, "plots")
    if not plot_path.is_dir():
        plot_path.mkdir()
    return plot_path


def vars_to_title(x, y):
    a = var_to_title(x)
    b = var_to_title(y)
    return a + " vs " + b


def vars_to_fname(x, y):
    return "sweep_" + x + "_vs_" + y + ".pdf"


def var_to_title(s):
    return s.replace('_', ' ').capitalize()


def plot_line(data, x, y, title, fname=None):
    fig = plt.figure(0)
    ax = sns.lineplot(x=x, y=y, data=data)
    ax.set_title(title)
    ax.set_xlabel(var_to_title(x))
    ax.set_ylabel(var_to_title(y))
    if fname:
        plt.savefig(fname)
    else:
        plt.show()
    plt.close(0)


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

