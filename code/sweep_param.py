import sys
import argparse
import numpy as np
import pandas as pd

# custom modules
import paths
import plots
from run_model import Experiment
from stats import Metrics

# GLOBAL VAR SETTINGS
NUM_REPETITIONS = 20    # repetitions per setting

# metrics to run per sweep
metrics = ["mean", "num_groups", "size_parity", "spread", 
           "coverage", "dispersion"]

def sweep_two_params(param1, param2, values1, values2):
    pass


def sweep_param(param, values, settings):
    # update paths in paths.py
    nlogo_path = paths.NLOGO_PATH_DEFAULT
    model_path = paths.MODEL_PATH_DEFAULT
    plot_path = plots.get_plot_dir(paths.OUTPUT_PATH_DEFAULT)


    # generate netlogo experiment settings file, run sims
    settings[param] = values

    # TODO: remove this.
    setup_file = "../netlogo/sample_experiments.txt"

    E = Experiment(model_path, name="sweep_"+param)
    #E = Experiment(model_path, name="test_1", setup_file=setup_file)
    E.generate_setup_file(settings=settings, repetitions=NUM_REPETITIONS)
    E.run_experiment()
    results = E.analyze_results(plot_distributions=False)
    E.write_results(results)

    for metric in metrics:
        plot_name = plots.vars_to_fname(param, metric)
        plot_title = plots.vars_to_title(param, metric)
        plot_fname = str(plot_path/plot_name)
        plots.plot_line(results, x=param, y=metric, title=plot_title, fname=plot_fname)

if __name__ == "__main__":
    settings = {
        "link-probability": [0.10],
        "population": [500],
        "threshold": [0.15],
        "media-1": ["false"],
        "media-2": ["false"],
        "media-1-bias": [0.5],
        "media-2-bias": [0.5],
    }
    sweep_param("population", [10, 20, 50, 100, 300, 500, 1000], settings)


