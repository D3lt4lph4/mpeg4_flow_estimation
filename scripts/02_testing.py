import os

# Remove tensorflow warning #
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

#############################

import argparse
from tqdm import tqdm
import numpy as np

from os import getcwd, listdir, makedirs
from os.path import join, isdir, splitext, basename

# Limit GPU usage
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# Import main dir to find the flow_estimation package
import sys
sys.path.append(getcwd())

from flow_estimation.factory import TestingConfiguration
from flow_estimation.factory import TestingConfigurationFactory

parser = argparse.ArgumentParser()
parser.add_argument("experiment", help="The experiment directory.")
parser.add_argument(
    "configuration", help="The testing configuration to evaluate against.")

args = parser.parse_args()

# Variable to hold the results
config_results = {}

# Create the directory where all the results will be output.
output_dir = join(args.experiment, "test", "test_{}".format(
    splitext(basename(args.configuration))[0]))

makedirs(output_dir, exist_ok=True)

# List the sub-experiments directories to be iterated upon
experiment_directories = sorted([join(args.experiment, experiment) for experiment in listdir(
    args.experiment) if isdir(join(args.experiment, experiment)) and experiment != "test"])

for experiment in experiment_directories:
    # Create the factory that will generate the configurations for evaluation
    configuration_factory = TestingConfigurationFactory(
        experiment, args.configuration)

    factory_iterator = iter(configuration_factory)

    # Create the current experiment output directory
    experiment_output_directory = join(
        output_dir, configuration_factory.experiment_id)

    makedirs(experiment_output_directory, exist_ok=True)

    # Initialize the results dict for the experiment
    if configuration_factory.experiment_id not in config_results:
        config_results[configuration_factory.experiment_id] = {
            "results": None}
    else:
        raise RuntimeError("Two experiment with the same id: {}".format(
            configuration_factory.experiment_id))

    # Iterate over the test configurations
    for config in tqdm(factory_iterator, desc="Processing the experiment {}".format(configuration_factory.experiment_id)):
        # Create the configuration object that will hold all we need to evaluate
        config_object = TestingConfiguration(config)

        config_object.prepare_runtime_stuff(experiment_output_directory)

        # Generate the graphs and evaluate the network with the current settings
        config_results[configuration_factory.experiment_id]["results"] = config_object.evaluator(**config_object.evaluation_parameters)

        config_results["results_metrics"] = config_object.evaluator.results_metrics

        # Cleaning all the tensorflow stuff
        config_object.clear()

        # Write the results to a file
        output_file = join(experiment_output_directory, "results.txt")
        with open(output_file, "w") as file:
            line = ""
            for idx, result in enumerate(config_results[configuration_factory.experiment_id]["results"]):
                line += "{}: {}\n".format(config_results["results_metrics"][idx], config_results[configuration_factory.experiment_id]["results"][idx])
            file.write(line)

# Write the results to files
output_file = join(output_dir, "results_avg.txt")

results_array = []
for experiment in config_results:
    if experiment == "results_metrics":
        continue
    
    results_array.append(config_results[experiment]["results"])

print(results_array)

mean = np.mean(results_array, axis=0)
std = np.std(results_array, axis=0)

with open(output_file, "w") as file:
    line = ""
    for idx, result in enumerate(mean):
        line += "{}: mean = {} ; std = {} \n".format(config_results["results_metrics"][idx], mean[idx], std[idx])
    file.write(line)