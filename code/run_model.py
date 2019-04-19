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
from pathlib import Path
from xml.etree import ElementTree as ET
import pandas as pd


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
        self.output_dir = Path(output_path).resolve()
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


    def analyze_results(self, csv_file=None):
        if csv_file is None:
            if self.output_file:
                csv_file = self.output_file
            else:
                print("ERROR: analyze needs results csv file")
                return

        # pandas won't work out-of-box b/c of weird format. 
        # need to parse by hand.
        #df = pd.read_csv(csv_file)
        #print(df.head())
        pass


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
            output_file = Path(self.output_dir, self.name + "_nlogo_out.csv").resolve()
        if not isinstance(output_file, Path):
            output_file = Path(output_file).resolve()

        # run the netlogo simulation
        print("Running experiment:\n  input file: {}\n  output file: {}\n" \
            .format(setup_file, output_file))
        nlogo_args = [ str(NLOGO_PATH_DEFAULT), 
            "--model", str(self.model),
            "--setup-file", str(setup_file),
            "--spreadsheet", str(output_file) ]
        res = subprocess.run(nlogo_args)

        # check that it worked
        if res.returncode != 0:
            print("Error, netlogo run failed")
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
        NLOGO_PATH_DEFAULT = args.nlogo_path

    model_arg = Path(args.model_path) if args.model_path else MODEL_PATH_DEFAULT
    setup_arg = Path(args.setup_file)
    output_arg = Path(args.output_path) if args.output_path else OUTPUT_PATH_DEFAULT

    E = Experiment(model_arg, name="test_1", output_path=output_arg, setup_file=setup_arg)
    E.run_experiment()
    E.analyze_results()


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
    
