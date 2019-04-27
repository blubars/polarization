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
NUM_REPETITIONS = 10    # repetitions per setting

# metrics to run per sweep
metrics = ["mean", "num_groups", "size_parity", "spread", "dispersion"]
#           "coverage"]


def sweep_two_params(p1, p2, settings):
    # update paths in paths.py
    nlogo_path = paths.NLOGO_PATH_DEFAULT
    model_path = paths.MODEL_PATH_DEFAULT
    plot_path = plots.get_plot_dir(paths.OUTPUT_PATH_DEFAULT)

    # generate netlogo experiment settings file, run sims
    param1 = p1["param"]
    param2 = p2["param"]
    values1 = sorted(list(set(p1["steps"] + p1["range"])))
    values2 = sorted(list(set(p2["steps"] + p2["range"])))
    settings[param1] = values1[:-1]
    settings[param2] = values2[:-1]

    E = Experiment(model_path, name="2d_sweep_"+param1+"_"+param2)
    E.generate_setup_file(settings=settings, repetitions=NUM_REPETITIONS)
    E.run_experiment()
    results = E.analyze_results(plot_distributions=False)
    E.write_results(results)
    #agg_res = E.aggregate_results(results, plot_distributions=False)

    # put aggregate data into plottable form.
    x = values1.copy()
    #x = [0] + x
    #x.append(2*x[-1]-x[-2]) # cheat to get around bad plot
    y = values2.copy()
    #y = [0] + y
    #y.append(2*y[-1]-y[-2])

    X, Y = np.meshgrid(x, y) # again, cheat...

    for metric in metrics:
        plot_title = plots.vars_to_title(param1, param2, metric)
        plot_fname = str(plot_path / plots.vars_to_fname(param1, param2, metric))
        z = np.zeros(X.shape).reshape(-1)
        for i,vals in enumerate(zip(X.ravel(), Y.ravel())):
            xx,yy = vals
            if xx > values1[-1] or yy > values2[-1]: # more cheating..
                z[i] = 0
            else:
                sel = (results[param1] == xx) & (results[param2] == yy)
                z[i] = results.loc[sel,metric].mean()
        #print("metric:{}, z:{}".format(metric,z))
        z = z.reshape(X.shape)
        plots.plot_2d_grid(x, y, z, param1, param2, title=plot_title, fname=plot_fname)


def sweep_param(param, values, settings):
    # update paths in paths.py
    nlogo_path = paths.NLOGO_PATH_DEFAULT
    model_path = paths.MODEL_PATH_DEFAULT
    plot_path = plots.get_plot_dir(paths.OUTPUT_PATH_DEFAULT)

    # generate netlogo experiment settings file, run sims
    settings[param] = values

    E = Experiment(model_path, name="sweep_"+param)
    #E = Experiment(model_path, name="test_1", setup_file=setup_file)
    E.generate_setup_file(settings=settings, repetitions=NUM_REPETITIONS)
    E.run_experiment()
    results = E.analyze_results(bins=50)
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
    #sweep_param("threshold", [10, 20, 50, 100, 300, 500, 1000], settings)

    # params to sweep. 
    # Note: max range is the max # on the plot list, so will not plot 
    # final param in step list if equal to max range.
    p1 = { "param":"threshold", "steps":[0.01, 0.02, 0.05, 0.075, 0.09, 0.1, 0.125, 0.15, 0.175, 0.2, 0.3, 0.4, 0.5], "range":[0,1] }
    p2 = { "param":"population", "steps":[100, 200, 300, 500], "range":[100,1000] }
    sweep_two_params(p1, p2, settings)

