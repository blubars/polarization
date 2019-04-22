import sys
import argparse
import numpy as np
import pandas as pd

# custom modules
import paths
import plots
from run_model import Experiment
from stats import Metrics

def sweep_param(param, values):
    # update paths in paths.py
    nlogo_path = paths.NLOGO_PATH_DEFAULT
    model_path = paths.MODEL_PATH_DEFAULT
    
    E = Experiment(, name="sweep_"+param)
    E.run_experiment()
    results = E.analyze_results(plot_distributions=False)
    E.write_results(results)
    agg_results = E.aggregate_results(plot_distributions=False)
    E.write_results(agg_results, name=E.name + "_agg_results.csv")

if __name__ == "__main__":
    sweep_param("population", [10, 20, 50, 100, 300, 500, 1000])


