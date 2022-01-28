import sys

import argparse

import gc

from shutil import copyfile

from os import listdir, environ, makedirs, getcwd
from os.path import join, basename, isdir, splitext

# Hiding the tensorflow warnings
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

sys.path.append(getcwd())
from flow_estimation.factory import TrainingConfigurationFactory, TrainingConfiguration

parser = argparse.ArgumentParser(
    description="Starts a training given a configuration file.")
parser.add_argument('configuration',
                    help="The json configuration file to use for the training.")
parser.set_defaults(feature=True)

args = parser.parse_args()

# Load the configuration file into the factory
configuration_factory = TrainingConfigurationFactory(args.configuration)

experiment_name = "_".join(
    splitext(basename(args.configuration))[0].split("_")[1:])

factory_iterator = iter(configuration_factory)

existing_dirs = sorted([directory for directory in listdir(environ["EXPERIMENTS_OUTPUT_DIRECTORY"]) if isdir(
    join(environ["EXPERIMENTS_OUTPUT_DIRECTORY"], directory))])

experiment_id = int(existing_dirs[-1].split("_")
                    [0]) + 1 if len(existing_dirs) > 0 else 1

# We iterate over the factory to get all of the training configurations objects
for config_dictionnary in factory_iterator:

    config = TrainingConfiguration(config_dictionnary)
    # Create the output directories for the current experiment
    output_dir = join(
        environ["EXPERIMENTS_OUTPUT_DIRECTORY"], "{:04d}_{}".format(experiment_id, experiment_name), config.sub_experiment_name)

    checkpoints_output_dir = join(output_dir, "checkpoints")
    config_output_dir = join(output_dir, "config")
    logs_output_dir = join(output_dir, "logs")

    # We create all the output directories
    makedirs(output_dir, exist_ok=True)
    makedirs(checkpoints_output_dir, exist_ok=True)
    makedirs(config_output_dir, exist_ok=True)
    makedirs(logs_output_dir, exist_ok=True)

    directories_dict = {"output": output_dir, "checkpoints_dir": checkpoints_output_dir,
                        "config_dir": config_output_dir, "log_dir": logs_output_dir}

    # Saving the starting weights if not None
    if config.weights is not None:
        copyfile(config.weights, join(checkpoints_output_dir, "source_weights.h5"))

    # Loading the model
    model = config.network

    config.prepare_runtime_callbacks(directories_dict)

    # We save the configuration files into the configuration directory (we
    # ask for a temp file for playing around if required)
    config.save(config_output_dir)

    print("Compiling the model ...")
    model.compile(loss=config.loss,
                  optimizer=config.optimizer,
                  metrics=config.metrics)

    print("Printing the model ...")
    model.summary()

    # We print in a file for easier checking of the model when required.
    with open(join(output_dir, 'model_summary.txt'), 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    print("Fit the model on the batches generated.")
    if "generator" in config.fit_parameters:
        model.fit_generator(**config.fit_parameters)
    else:
        model.fit(**config.fit_parameters)

    print("Done with the training, cleaning up and proceding with the next training...")

    # Cleaning up and trying to avoid data accumulation
    config.clear()
    del config
    gc.collect()

print("Done training.")
