import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns

from collections import namedtuple

Group = namedtuple("Group", ["min", "max", "count"])

datas_real = [ [
    0.10023941607513752, 0.10029988008885937, 0.10031314135570898, 
    0.1003336101451271, 0.10033484179915522, 0.10034501218029417, 
    .100350421788459, 0.10035286397079722, 0.10036556686197377, 
    .10037491275655894, 0.10037787139530455, 0.10037851662782618, 
    .10038153216498806, 0.10038777690098075, 0.10039651573393941, 
    .10040018357354837, 0.10040057779381535, 0.10040344520807166, 
    .10041220028147833, 0.1004123091393069, 0.10041373414632213, 
    .10042124834303896, 0.418089287451237, 0.418089322689329, 
    .4180893536400485, 0.4180893872676364, 0.4180894139695995, 
    .4180894920376869, 0.4180895052629091, 0.4180895326505157, 
    .41808957179710265, 0.4180895794584438, 0.41808958019399983, 
    .4180895969610674, 0.4180896072505537, 0.4180896127518728, 
    .41808962515587134, 0.4180896472734665, 0.41808965540858567, 
    .418089681003653, 0.41808968534255936, 0.4180897152733609, 
    .4180897503499411, 0.4180897631201795, 0.4180897688103656, 
    .4180897771133891, 0.41808983847369824, 0.41808986755711547, 
    .4180898901015791, 0.4180898979285596, 0.41808989956312065, 
    .41808994090765433, 0.41808994846804404, 0.4180899537226777, 
    .4180900863297739, 0.4180901405583624, 0.41809018368661316, 
    .4180902368959316, 0.4180904819474387, 0.418090596799493, 
    .8973178563964178, 0.89734944396729, 0.8974052291451833, 
    .8974074792319414, 0.8974081531972216, 0.8974687468450631, 
    .8974958851884431, 0.897502974462471, 0.8975040585671481, 
    .8975333720375329, 0.8975471342614022, 0.8975516201726111, 
    .8975531707424516, 0.8975547836280646, 0.8975729548396675, 
    .8975832775207747, 0.8975834393960809, 0.8975844692428048, 
    .8975980238976237, 0.8975994320598889, 0.897605498753089, 
    .8976199549970936, 0.8976473316581983, 0.8976488274847513, 
    .8976536287408721, 0.8976611219239843, 0.8976812048755042, 
    .897684514832682, 0.8977028020337441, 0.8977077155138827, 
    .8977190794853169, 0.8977701927170028, 0.8977752992863457, 
    .8978379043073025, 0.8978654177114163, 0.8978744087318089, 
    .8978918611469672, 0.8978967382892599, 0.8979003223017609, 
    .8980281141121026], [
    0.10427432519351156, 0.10428145147711633, 0.10428453394465878, 
    0.10428676864991183, 0.10429000261061552, 0.1043481582076999, 
    .10437662219628517, 0.10439522365230898, 0.104398563187787, 
    .10439905417152075, 0.10440460887694603, 0.10441332467751223, 
    .10442053739564638, 0.10442904585049029, 0.1044419123476063, 
    .10445401621376578, 0.10445853052871484, 0.10446582388647302, 
    .10450481882722899, 0.10451002541009731, 0.10451485583441053, 
    .10452239611213192, 0.10454476438740362, 0.10455136988260026, 
    .10456090544223688, 0.1045611637129553, 0.10456190215508179, 
    .10456412201105464, 0.10457202858243299, 0.10457368109899855, 
    .10457539084661004, 0.10457684088900783, 0.10458706108047457, 
    .10459301292673993, 0.10460692489688839, 0.10461875339709054, 
    .10462327817548694, 0.10464465820169776, 0.10464562414093596, 
    .10465720613368217, 0.1046744170300768, 0.1046851874869304, 
    .10468556291418504, 0.10468909116236111, 0.10469561249484009, 
    .10472551882001176, 0.10474018258251212, 0.1047455388639906, 
    .10474916259479887, 0.10474934810363118, 0.1047616225445152, 
    .10476607054410723, 0.10479105995661675, 0.1047975323678512, 
    .10481749335662283, 0.10482655986287756, 0.1048365599948135, 
    .10483783120191977, 0.10485522145476363, 0.8980772046061314, 
    .8980940791036615, 0.8981304144009916, 0.8981328271442445, 
    .8981334026608985, 0.8981454008384853, 0.8981523305312649, 
    .8981673091220471, 0.8981726115722463, 0.8981744895149731, 
    .898187070113304, 0.8981922903214014, 0.898199172933497, 
    .8982010180104949, 0.8982037438382809, 0.8982043283137183, 
    .8982199008694608, 0.8982281391863718, 0.8982311069411532, 
    .8982528818137737, 0.8982560239935279, 0.8982601324954168, 
    .8982807325638266, 0.8982808013712479, 0.8982845506322433, 
    .8982870109307076, 0.8983018361106094, 0.8983094498557976, 
    .8983239502085757, 0.8983279742596793, 0.8983314289103996, 
    .8983386813224958, 0.8983392896692688, 0.8983399964044954, 
    .8983413637384717, 0.8983476276650632, 0.8983519590145946, 
    .8983775814079714, 0.8983779088868475, 0.89843339134954, 
    .8984752456430136], [
    0.10018895137544775, 0.10020109344238302, 0.10020411339417748, 
    0.10020508563145358, 0.10020684931614862, 0.10020827932055866, 
    .10021154614643439, 0.10021232332773608, 0.1002132919574761, 
    .10021489148471764, 0.10021551010092998, 0.1002197072790138, 
    .10022059342116427, 0.10022268272089044, 0.10022307216601155, 
    .10022399432994834, 0.10022711013193336, 0.10022721545979846, 
    .10022739239103848, 0.1002327783309058, 0.10023510264153454, 
    .10023554211828505, 0.10023679624346811, 0.10023982721328559, 
    .10023995809721134, 0.10024098011877802, 0.1002426792227395, 
    .10024463308671679, 0.10024523644648098, 0.342584375053548, 
    .5709548220306814, 0.5709549443616001, 0.5709550186120596, 
    .5709551559441934, 0.5709552067126537, 0.5709552089458957, 
    .5709552341125321, 0.5709552687320358, 0.5709553056654907, 
    .5709553278048756, 0.5709553999416823, 0.5709554044654612, 
    .570955405431072, 0.5709554651619249, 0.5709555128837915, 
    .5709555856545881, 0.5709555887007473, 0.5709555890794611, 
    .5709556017913682, 0.5709556036499037, 0.5709556362466057, 
    .5709557081740103, 0.570955710730225, 0.8959963118223024, 
    .8960769866052032, 0.8961102958237371, 0.896122446126489, 
    .8961284059091175, 0.8961402358854806, 0.8961532494087942, 
    .8961677584100339, 0.8961697213612293, 0.8961712800058381, 
    .8961850197629958, 0.8962009885208101, 0.8962114930983585, 
    .8962250702225338, 0.896230548710019, 0.8962349752155284, 
    .896236717454727, 0.8962453853181386, 0.896246478646494, 
    .8962780949422312, 0.8962900972803175, 0.8962979168968348, 
    .8963181714869473, 0.8963243880314535, 0.8963301662808306, 
    .8963399931462624, 0.8963406034495799, 0.8963816165572643, 
    .8963839712935718, 0.8963858388336248, 0.8964025962500601, 
    .8964193256126084, 0.8964217791013942, 0.896430676587288, 
    .8964602857105654, 0.8964805082548316, 0.8964878951075107, 
    .8964914909238336, 0.8965144014814932, 0.896517647545787, 
    .8965305392639191, 0.896551253718872, 0.8965716500076312, 
    .8966102670312004, 0.89661796469883, 0.8966694184587982, 
    .8966902472870887] ]

def normalize(a):
    total = sum(a)
    return [aa / total for aa in a]


class Metrics:
    def __init__(self, data):
        self.data = np.around(data, decimals=2)
        self.get_groups()

    def spread(self):
        """ also called range... highest - lowest.
            drawback: doesn't consider if the most extreme are simply 
            outliers, or are representative. """
        return max(self.data) - min(self.data)

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
        fig = plt.figure(0)
        bins = np.linspace(0, 1, num_bins+1)
        ax = sns.distplot(self.data, bins=bins, rug=True, kde=kde)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        if name:
            plt.savefig(name)
        else:
            plt.show()
        fig.close(0)

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
        metrics["coverage"] = self.coverage()
        metrics["num_groups"] = self.num_groups()
        metrics["dispersion"] = self.dispersion()
        metrics["distinctness"] = self.distinctness()
        metrics["size_parity"] = self.size_parity()
        metrics["group_consensus"] = self.group_consensus()
        return metrics


# distinctness, divergence, group concensus
