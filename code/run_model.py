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

import sys
import subprocess
import argparse
import csv
from pathlib import Path
from xml.etree import ElementTree as ET
import pandas as pd
from stats import Metrics


# GLOBALS :/
NLOGO_PATH_DEFAULT = Path(Path.home(), "NetLogo/netlogo-headless.sh")
MODEL_PATH_DEFAULT = Path("../netlogo/polarization.nlogo")
OUTPUT_PATH_DEFAULT = Path("../experiments/")


class Experiment:
    """ set up, run, and analyze experiment.
        PARAMS:
        - model: path to netlogo model to run
        - name: name of experiment (used to name files)
        - setup_file: netlogo experimental XML file 
        - output_path: directory to store results """
    def __init__(self, model, name=None, output_path=OUTPUT_PATH_DEFAULT, setup_file=None):
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
        with open(csv_file) as csv_file:
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

    def write_results(self, results_df, output_dir=None):
        """ write dataframe to csv """
        if not output_dir:
            output_dir = self.output_dir
        output_file = output_dir / (self.name + "_results.csv")
        results_df.to_csv(path_or_buf=output_file, index=False)
        print("  * Results written to {}".format(output_file))

    def analyze_results(self, csv_file=None):
        if csv_file is None:
            if self.output_file:
                csv_file = self.output_file
            else:
                print("ERROR: analyze needs results csv file")
                return
        df = self.parse_results(csv_file)

        print("  Results:")
        results = {}
        for ix,run in df.iterrows():
            # each row in the dataframe is one run, with a given
            # set of settings. analyze it.
            beliefs = run["final_values"]
            analyzer = Metrics(beliefs)
            metrics = analyzer.run_all()
            print("   [{}]: G({},{}) --> num_groups:{}, spread={}" \
                .format(ix, run["population"], run["threshold"], 
                        metrics["num_groups"], metrics["coverage"]))
            # aggregate into list of values for each metric
            for metric,val in metrics.items():
                if metric in results:
                    results[metric].append(val)
                else:
                    results[metric] = [val]
            # plot? 
            plot_name = self.name + str(ix) + ".pdf"
            plot_title = "Final Distribution\n(Pop:{}, thresh:{})".format(run["population"], run["threshold"])
            #analyzer.plot(title=plot_title, name=plot_name)
        df_result = pd.DataFrame(results)
        return pd.concat([df.drop("final_values", axis=1), df_result], axis=1) 


    def generate_setup_file(self, name, settings, repetitions, path=None):
        """ generate a netlogo experimental setup XML file """
        if not path:
            path = self.output_dir

        # set up experiment netlogo global values
        exp_attribs = {
            "name": name,
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
        outpath = Path(path, name + ".xml")
        print("Writing settings to file: {}".format(str(outpath)))
        et = ET.ElementTree(tree_root)
        et.write(str(outpath))
        self.setup_file = outpath


    def run_experiment(self, setup_file=None, output_file=None):
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
            output_file = Path(self.output_dir, self.name + "_nlogo_out.csv").resolve(strict=False)
        if not isinstance(output_file, Path):
            output_file = Path(output_file).resolve(strict=False)

        print("Running experiment:\n  * input file: {}\n  * output file: {}" \
            .format(setup_file, output_file))

        # check if output file already exists. don't re-run if it does!
        if output_file.exists():
            print("  * NOTE: Result of experiment {} already run, skipping.\n    -- see file: {}".format(self.name, output_file))
            self.output_file = output_file
            return success

        # run the netlogo simulation
        nlogo_args = [ str(NLOGO_PATH_DEFAULT), 
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
        NLOGO_PATH_DEFAULT = Path(args.nlogo_path).resolve()

    model_arg = Path(args.model_path) if args.model_path else MODEL_PATH_DEFAULT
    setup_arg = Path(args.setup_file)
    output_arg = Path(args.output_path) if args.output_path else OUTPUT_PATH_DEFAULT

    E = Experiment(model_arg, name="test_1", output_path=output_arg, setup_file=setup_arg)
    E.run_experiment()
    results = E.analyze_results()
    E.write_results(results)

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
    
    E = Experiment(MODEL_PATH_DEFAULT, name="test_2", output_path=OUTPUT_PATH_DEFAULT)
    E.generate_setup_file("test_gen_2", settings=exp_2_settings, repetitions=5)
    E.run_experiment()
    E.analyze_results()
"""
    
