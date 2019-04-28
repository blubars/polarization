#!/usr/bin/env python3

# just trying things out for now.
# to use: need to set default paths in GLOBALS section below, or
#   pass those in through the cmd-line args.

# note: an experimental setup dictionary should look like this:
"""
experiment_dict = {
    "link-probability": [0.2],
    "population": [100, 300, 500],
    "threshold": [0.15],
    "media-1": ["true"],
    "media-2": ["true"],
    "media-1-bias": [0.1],
    "media-2-bias": [0.9],
}
"""

# system modules
import sys
import subprocess
import argparse
import csv
import numpy as np
from pathlib import Path
from xml.etree import ElementTree as ET
import pandas as pd

# local modules
import paths
import plots
from stats import Metrics


#global paths.NLOGO_PATH_DEFAULT
#global paths.MODEL_PATH_DEFAULT
#global paths.OUTPUT_PATH_DEFAULT

class Experiment:
    """ set up, run, and analyze experiment.
        PARAMS:
        - model: path to netlogo model to run
        - name: name of experiment (used to name files)
        - setup_file: netlogo experimental XML file 
        - output_path: directory to store results """
    def __init__(self, model, name=None, output_path=paths.OUTPUT_PATH_DEFAULT, setup_file=None):
        self.model = Path(model).resolve()
        self.output_dir = Path(output_path)
        self.setup_file = Path(setup_file).resolve() if setup_file else None
        self.setup_file = setup_file
        self.output_file = None
        self.results = None
        self.name = name
        if self.name is None and not self.setup_file is None:
            # if not set, try to set name from input setup file
            self.name = setup_file.stem 
        if not self.output_dir.is_dir():
            print("Creating output directory {}".format(self.output_dir))
            self.output_dir.mkdir(parents=True)


    def parse_results(self, csv_file):
        """ returns a pandas dataframe with parsed results """
        # resolve CSV file name
        if csv_file is None:
            if self.output_file:
                csv_file = self.output_file
            else:
                print("  * ERROR: analyze needs results csv file")
                return

        # do parsing!
        valid_rows = {
            "[run number]":int,
            "link-probability":float,
            "media-2-bias":float,
            "media-1-bias":float,
            "population":int,
            "media-1":bool,
            "threshold":float,
            "media-2":bool,
            "[steps]":int
        }

        output_dict = {}
        with open(str(csv_file)) as csv_file:
            final_values_row = 0
            for row in csv.reader(csv_file, delimiter=','):
                if len(row) > 0:
                    # experimental settings rows
                    key = row[0]
                    if key in valid_rows:
                        name = key.strip("[]").replace(" ", "-")
                        values=[]
                        for value in row[1:]:
                            # possible types are bool, int, float.
                            # convert to float if possible.
                            convert = valid_rows[key]
                            if convert is bool:
                                convert = lambda x: True if x == "true" else False
                            values.append(convert(value))
                        output_dict[name]=values

                    # experimental results row
                    elif final_values_row > 0:
                        if final_values_row == 2:
                            final_vals=[]
                            for values in row[1:]:
                                values=values.strip("[]").split(' ')
                                float_values=[float(value) for value in values]
                                final_vals.append(float_values)
                            output_dict['final_values']=final_vals
                        final_values_row += 1

                elif not final_values_row:
                    # skip two garbage rows btw settings and results
                    final_values_row=1
        return pd.DataFrame(output_dict)

    def write_results(self, results_df, name=None, output_dir=None):
        """ write dataframe to csv """
        if not output_dir:
            output_dir = self.output_dir
        if not name:
            name = (self.name + "_results.csv")
        output_file = output_dir / name
        results_df.to_csv(path_or_buf=output_file, index=False)
        print("  * Results written to {}".format(output_file))


    def aggregate_group(self, x):
        # x is a series (within a group).
        if x.dtype == "object":
            a = np.array([_ for _ in x])
            return np.mean(a, axis=0).ravel().tolist()
        else:
            # return first value
            return x.values[0]


    def aggregate_results(self, results, plot_distributions=False):
        """ if runs use the same settings, combine them (mean, std, count) """
        # make a new index to combine off of, combo of all settings.
        setting_cols = ["link-probability", "media-2-bias", "media-1-bias",
                "population", "media-1", "threshold", "media-2"]
        numeric_cols = ["steps", "mean", "num_groups", "size_parity", "spread", 
                "coverage", "dispersion", "group_consensus"]
        settings = results.loc[:,setting_cols]
        keys = []
        for ix,row in settings.iterrows():
            key = "".join([str(cell) + '_' for cell in row])
            keys.append(key)
        df = pd.concat([results, pd.DataFrame(keys, columns=["key"])], axis=1)

        # group by new index, run aggregation functions
        grouped = df.groupby(by="key")
        group_settings = grouped[setting_cols + ["avg_degree"]].agg("first")
        group_metrics = grouped[numeric_cols].agg(["mean", "std"])
        hist_agg_func = lambda x: np.mean(np.array([_ for _ in x]), axis=0).ravel().tolist()
        group_hist = grouped["histogram"].agg(hist_agg_func)
        group_size = grouped["run-number"].agg("size").rename("support")
        results = pd.concat([group_settings, group_metrics, group_hist, group_size], axis=1)

        # flatten column names
        new_cols = []
        for ci, col in enumerate(results.columns):
            if not isinstance(col, str):
                col = col[0] + '_' + col[1]
            new_cols.append(col) 
        results.columns = new_cols

        # plot grouped results, if desired
        if plot_distributions:
            for ig,group in results.iterrows():
                plot_name = self.name + "_{}_agg_distr.png".format(ig)
                plot_path = Path(self.output_dir, "plots")
                if not plot_path.is_dir():
                    plot_path.mkdir()
                #plot_title = "Final Belief PDF\n(thresh:{}, media:{})".format(group["threshold"], group["media-1-bias"])
                plot_title = "Final Belief PDF\n(thresh:{}, avg # groups:{})".format(group["threshold"], round(float(group["num_groups_mean"]), 2))
                bins = np.linspace(0, 1, 21)
                plots.plot_bar(group["histogram"], bins, title=plot_title, name=str(plot_path / plot_name), norm=group["population"])
                #plots.plot_dist(group["histogram"], num_bins=20, title=plot_title, name=str(plot_path / plot_name))
        return results

    def analyze_results(self, csv_file=None, plot_distributions=False, bins=20):
        """ parse CSV file and collect polarization metrics on each run """
        if csv_file is None:
            if self.output_file:
                csv_file = self.output_file
            else:
                print("ERROR: analyze needs results csv file")
                return
        df = self.parse_results(csv_file)
        # add avg degree for given settings.
        df['avg_degree'] = (df['population'] - 1) * df['link-probability']

        print("  Results:")
        results = {}
        for ix,run in df.iterrows():
            # each row in the dataframe is one run, with a given
            # set of settings. analyze it.
            beliefs = run["final_values"]
            analyzer = Metrics(beliefs, bins=bins)
            metrics = analyzer.run_all()
            print("   [{}]: G({},{}) --> num_groups:{}, spread={}" \
                .format(ix, run["population"], run["threshold"], 
                        metrics["num_groups"], metrics["spread"]))
            # aggregate into list of values for each metric
            for metric,val in metrics.items():
                if metric in results:
                    results[metric].append(val)
                else:
                    results[metric] = [val]
            # plot
            if plot_distributions:
                plot_name = self.name + "_distr_{}.pdf".format(str(ix))
                plot_path = Path(self.output_dir, "plots")
                if not plot_path.is_dir():
                    plot_path.mkdir()
                plot_title = "Final Distribution\n(Pop:{}, thresh:{})".format(run["population"], run["threshold"])
                analyzer.plot(title=plot_title, name=str(plot_path / plot_name))
        df_result = pd.DataFrame(results)
        return pd.concat([df.drop("final_values", axis=1), df_result], axis=1)


    def generate_setup_file(self, settings, repetitions, path=None):
        """ generate a netlogo experimental setup XML file """
        if not path:
            path = self.output_dir

        # set up experiment netlogo global values
        exp_attribs = {
            "name": self.name,
            "repetitions": str(repetitions),
            "runMetricsEveryStep": "false"
        }
        exp_root = ET.Element("experiment", attrib=exp_attribs)
        setup = ET.Element("setup")
        setup.text = "setup"
        go = ET.Element("go")
        go.text = "go"
        metric = ET.Element("metric")
        metric.text = "sort [belief] of people"
        elems = [setup, go, metric]

        # set up global variable initializations
        for setting,values in settings.items():
            setting_e = ET.Element("enumeratedValueSet", attrib={"variable":setting})
            for val in values:
                val_e = ET.Element("value", attrib={"value":str(val)})
                setting_e.append(val_e)
            elems.append(setting_e)
        exp_root.extend(elems)

        # Experiment is actually child of top-level Exeriments tag
        tree_root = ET.Element("experiments")
        tree_root.append(exp_root)

        # write experiment xml file to path
        outpath = Path(path, self.name + ".xml")
        print("Writing settings to file: {}".format(str(outpath)))
        et = ET.ElementTree(tree_root)
        et.write(str(outpath))
        self.setup_file = outpath


    def run_experiment(self, setup_file=None, output_file=None, nlogo_headless=paths.NLOGO_PATH_DEFAULT):
        """ run experiment. returns 0 for failure, 1 for success """
        success = True
        if not setup_file:
            if self.setup_file:
                setup_file = self.setup_file
            else:
                print("ERROR: run_experiment needs setup file")
                success = False
        if not isinstance(setup_file, Path):
            setup_file = Path(setup_file).resolve()
        if not output_file:
            output_file = Path(self.output_dir, self.name + "_nlogo_out.csv")
        if not isinstance(output_file, Path):
            output_file = Path(output_file)

        print("Running experiment:\n  * input file: {}\n  * output file: {}" \
            .format(setup_file, output_file))

        # check if output file already exists. don't re-run if it does!
        if output_file.exists():
            print("  * NOTE: Result of experiment {} already run, skipping.\n    -- see file: {}".format(self.name, output_file))
            self.output_file = output_file
            return success

        # run the netlogo simulation
        nlogo_args = [ str(nlogo_headless), 
            "--model", str(self.model),
            "--setup-file", str(setup_file),
            "--spreadsheet", str(output_file) ]
        res = subprocess.run(nlogo_args)

        # check that it worked
        if res.returncode != 0:
            print("  * Error, netlogo run failed")
            print(res)
            output_file = None
            success = False
        if res.stdout:
            print(res.stdout)
        self.output_file = output_file
        return success


def main():
    """ run in command-line mode. """
    parser = argparse.ArgumentParser(description="Runs a headless netlogo model")
    parser.add_argument("setup_file", help="Experiment setup input file")
    parser.add_argument("--output_path", help="Path to store result files")
    parser.add_argument("--nlogo_path", help="Path to netlogo folder")
    parser.add_argument("--model_path", help="Path to netlogo model file")
    args = parser.parse_args()
    if args.nlogo_path:
        paths.NLOGO_PATH_DEFAULT = Path(args.nlogo_path, "netlogo-headless.sh").resolve()

    model_arg = Path(args.model_path) if args.model_path else paths.MODEL_PATH_DEFAULT
    setup_arg = Path(args.setup_file)
    output_arg = Path(args.output_path) if args.output_path else paths.OUTPUT_PATH_DEFAULT

    E = Experiment(model_arg, name="test_1", output_path=output_arg, setup_file=setup_arg)
    E.run_experiment()
    results = E.analyze_results(plot_distributions=False)
    E.write_results(results)
    agg_results = E.aggregate_results(results, plot_distributions=False)
    E.write_results(agg_results, name=E.name + "_agg_results.csv")

if __name__ == "__main__":
    main()

"""
    exp_2_settings = { # dictionary of experiment settings
        "link-probability": [0.18],
        "population": [100, 300, 500],
        "threshold": [0.15],
        "media-1": ["true"],
        "media-2": ["false"],
        "media-1-bias": [0.1],
        "media-2-bias": [0.5],
    }

    E = Experiment(paths.MODEL_PATH_DEFAULT, name="test_2", output_path=paths.OUTPUT_PATH_DEFAULT)
    E.generate_setup_file(settings=exp_2_settings, repetitions=5)
    E.run_experiment()
    E.analyze_results()
"""

