import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from collections import namedtuple
import plots

Group = namedtuple("Group", ["min", "max", "count"])

def normalize(a):
    total = sum(a)
    return [aa / total for aa in a]


class Metrics:
    def __init__(self, data, bins=20):
        self.data = np.around(data, decimals=2)
        self.num_bins=bins
        self.get_groups(self.num_bins)

    def bin(self, num_bins=20):
        bins = np.linspace(0,1,num_bins+1)
        counts, bins = np.histogram(self.data, bins=bins)
        return counts

    def spread(self):
        """ also called range... highest - lowest.
            drawback: doesn't consider if the most extreme are simply 
            outliers, or are representative. """
        mh = max(self.data)
        ml = min(self.data)
        return (mh - ml)

    def mean(self):
        return np.mean(self.data)

    def coverage(self, num_bins=20):
        """  percent of the spectrum between 0, 1 that is occupied by
             at least one person """
        counts, bins = np.histogram(self.data, bins=num_bins)
        return len(list(filter(lambda x: x > 0, counts))) / num_bins

    def dispersion(self, base=2, metric="entropy"):
        """ statistical variation, e.g: mean diff, avg abs deviation, stdev, entropy. 
            consders overall shape of distribution """
        if metric == "entropy":
            # let's use entropy!
            value, counts = np.unique(self.data, return_counts=True)
            return entropy(counts, base=base)
        if metric == "std":
            return np.std(self.data)

    def plot(self, num_bins=20, title="", xlabel="", name=None, kde=False):
        plots.plot_dist(self.data, num_bins=num_bins, title=title, xlabel=xlabel, name=name, kde=kde)

    def get_groups(self, num_bins=20, data_range=[0, 1]):
        """ find endogenously-defined groups.
            return list of group tuples, defining range and 
            count of group membership """
        groups = []
        bins = np.linspace(data_range[0], data_range[1], num_bins+1)
        group_min, group_max, group_count = None, None, None
        group_start = None
        #counts, bins = np.histogram(data, bins=num_bins)
        for bin_i in range(len(bins)-1):
            left, right = bins[bin_i], bins[bin_i+1]
            if right == bins[-1]:
                # last bin: less than or equal to.
                sel = np.logical_and(self.data>=left, self.data<=right)
            else:
                sel = np.logical_and(self.data>=left, self.data<right)
            vals = self.data[sel]

            # start a new bin, continue existing, or finish?
            if group_start is None and len(vals) > 0:
                # start a new bin
                group_start = bin_i
                group_min, group_max, group_count = 100, 0, 0

            if not (group_start is None):
                # continuing bin. are we done?
                if len(vals) != 0:
                    # update min/max w/ current bin.
                    bin_min, bin_max = min(vals), max(vals)
                    if bin_min < group_min: group_min = bin_min
                    if bin_max > group_max: group_max = bin_max
                    group_count += len(vals)

                # done.
                if len(vals) == 0 or right == bins[-1]:
                    g = Group(group_min, group_max, group_count)
                    group_start = None
                    groups.append(g)
        self.groups = groups
        return groups

    def get_group_values(self, group):
        sel = np.logical_and(self.data>=group.min, self.data<=group.max)
        return self.data[sel]

    def pairwise_group_generator(self):
        for i in range(len(self.groups)-1):
            for j in range(i+1, len(self.groups)):
                yield (self.groups[i], self.groups[j])

    def num_groups(self):
        return len(self.groups)

    def distinctness(self, groups=None):
        """ group-based metric. sort of how "bimodal" the groups are?
        Different for endogenous vs exo- defined groups.
            if endogenous: height at minimum of overlap between the kde 
                distributions. note: always 0 for this experiment, since we
                run until convergence.
            if exogenous: amount of overlap between the group distributions
        """
        # TODO: exogenous version
        return 0


    def group_divergence(self):
        """ a measure that makes most sense with bimodality.
            captures how distant average group ideals are, by using
            the difference between the means """
        total_div = 0
        num_pairs = 0
        for g1,g2 in self.pairwise_group_generator:
            m1 = np.mean(self.get_group_values(g1))
            m2 = np.mean(self.get_group_values(g2))
            total_div += abs(m1 - m2)
        return total_div / num_pairs

    def size_parity(self, method="avgdiff", base=2):
        # normalize sizes
        sizes = [g.count for g in self.groups]
        if method == "avgdiff":
            sizes = normalize(sizes)
            avg_size = np.mean(sizes)
            avg_dist = 0
            for size in sizes:
                avg_dist += abs(avg_size - size)
            return avg_dist / len(sizes)
        else: # entropy:
            return entropy(sizes, base=base)

    def group_consensus(self):
        # median absolute deviation, averaged over all groups
        total_mads = 0
        for g in self.groups:
            vals = self.get_group_values(g)
            med = np.median(vals)
            mad = np.median(np.abs(vals - med))
            total_mads += mad
        return total_mads / len(self.groups)


    def run_all(self):
        """ return a dict of all metrics """
        metrics = {}
        metrics["mean"] = self.mean()
        metrics["spread"] = self.spread()
        metrics["coverage"] = self.coverage(num_bins=self.num_bins)
        metrics["num_groups"] = self.num_groups()
        metrics["dispersion"] = self.dispersion()
        #metrics["distinctness"] = self.distinctness()
        metrics["size_parity"] = self.size_parity()
        metrics["group_consensus"] = self.group_consensus()
        metrics["histogram"] = self.bin(num_bins=self.num_bins)
        return metrics


