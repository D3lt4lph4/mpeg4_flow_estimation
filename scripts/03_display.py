from os.path import join

import argparse
import sys
import json

#### Remove all warnings ####

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#############################

from os import getcwd

sys.path.append(getcwd())

from flow_estimation.factory import DisplayConfiguration

parser = argparse.ArgumentParser()

parser.add_argument("experiment", help="The experiment directory.", type=str)
parser.add_argument("weights", help="The weights to load.", type=str)
parser.add_argument("config_set_file", help="The set file to be used to load the data.", type=str)
parser.add_argument('-gt', '--groundTruth', action='store_true')
parser.add_argument('-ns',
                    '--number_of_samples',
                    help="The number of samples to display.",
                    default=4,
                    type=int)
parser.add_argument("-o",
                    "--output",
                    help="Where the prediction should be output.",
                    default=None)
parser.add_argument("-s",
                    "--shuffle",
                    help="If the data should be shuffled before display",
                    action="store_true")
args = parser.parse_args()

# Read the configuration file from the experiment folder
configuration_file = join(args.experiment, "config/config.json")

# Get the configuration file and load the everything we need for display
with open(configuration_file) as json_file:
    configuration_file = json.load(json_file)

# Add the set file to use for testing to the test generator
configuration_file["generator"]["test"]["configuration"]["set_file"] = args.config_set_file

config = DisplayConfiguration(configuration_file)

# Loading the model
model = config.network
model.load_weights(args.weights)

# Select the available generator
if config.test_generator:
    generator = config.test_generator
elif config.validation_generator:
    generator = config.validation_generator
else:
    generator = config.train_generator

# If the ground truth should be displayed or not along the predictions
if args.groundTruth:
    config.displayer.display_with_gt(model, generator, args.number_of_samples,
                                     args.output, args.shuffle)
else:
    config.displayer.display(model, generator, args.number_of_samples,
                             args.output, args.shuffle)
