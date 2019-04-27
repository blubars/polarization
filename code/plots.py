from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_3d(x, y, z, xlabel, ylabel, zlabel, title, fname):
    fig = plt.figure(0)
    fig.suptitle(title)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x, y, z, alpha=0.9, cmap=plt.cm.coolwarm, linewidth=2, antialiased=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.savefig(fname)
    plt.close(0)


def plot_2d_grid(x, y, z, xlabel, ylabel, title, fname=None):
    """ 2d grid pixel plot """
    X, Y = np.meshgrid(x, y)
    fig = plt.figure(0)
    ax = plt.gca()
    #ax.set_xlim(min(x), max(x))
    #ax.set_ylim(min(y), max(y))
    #ax.axis([min(x), max(x), min(y), max(y)])

    X, Y = np.meshgrid(x, y)
    #plt.imshow(z, extent=(min(x), max(x), max(y), min(y)),interpolation='none')
    plt.pcolormesh(X, Y, z) #, shading="gouraud")
    plt.colorbar()
    ax.set_title(title)
    ax.set_xlabel(var_to_title(xlabel))
    ax.set_ylabel(var_to_title(ylabel))
    if fname:
        plt.savefig(fname)
    else:
        plt.show()
    plt.tight_layout()
    plt.close(0)

    #fig.suptitle("MVR Violations Over $n$ and $p$ for $G(n,p)$ Random Graphs")
    #ax = fig.add_subplot(111, projection='3d')
    #X = df["n"]
    #Y = df["p"]
    #z = df["v"].values
    #ax.plot_trisurf(X, Y, z, alpha=0.9, cmap=plt.cm.coolwarm, linewidth=2, antialiased=True)
    #ax.set_ylabel("$p$")
    #ax.set_xlabel("Number of nodes ($n$)")
    #ax.set_zlabel("Violations ($V$)\nfor Best Found MVR")
    #plt.show()
    #plt.savefig("part1b.pdf")
    #plt.close(0)


def get_plot_dir(output_dir):
    plot_path = Path(output_dir, "plots")
    if not plot_path.is_dir():
        plot_path.mkdir(parents=True)
    return plot_path


def vars_to_title(x, y, z=None):
    a = var_to_title(x)
    b = var_to_title(y)
    if z:
        c = var_to_title(z)
        return "{} vs {} and {}".format(c, a, b)
    else:
        return a + " vs " + b


def vars_to_fname(x, y, z=None):
    if z:
        return "2d_sweep_" + x + "_vs_" + y + "_vs_" + z + ".pdf"
    else:
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

