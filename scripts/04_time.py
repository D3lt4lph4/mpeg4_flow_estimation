import sys

from os import getcwd

import numpy as np

sys.path.append(getcwd())

import time

from tqdm import tqdm

import argparse
import json

from flow_estimation import networks, generators
from flow_estimation import models
# from flow_estimation import trackers

import tensorflow as tf

# import nvidia_smi

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)


# nvidia_smi.nvmlInit()
# handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
# mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

# original_memory = mem_res.used / (1024**2)

parser = argparse.ArgumentParser(
    description="Script to calculate the computation time for a given network."
)

parser.add_argument("configuration",
                    help="The configuration file to use for the estimation.",
                    type=str)
parser.add_argument("-nb", "--number_of_batches", help="The number of time the prediction should be run on the batch of data.", type=int, default=1000)

args = parser.parse_args()

number_of_samples = args.number_of_batches

with open(args.configuration, "r") as json_file:
    config_dict = json.load(json_file)

# Create the generator
generator_configuration = config_dict["generator"]["configuration"]
generator = getattr(
    generators, config_dict["generator"]["name"])(**generator_configuration)

batch_size = config_dict["generator"]["configuration"]["batch_size"]

# Load the network
network_configuration = config_dict["network"]["configuration"]
network = getattr(networks,
                  config_dict["network"]["name"])(**network_configuration)

if config_dict["weights"] is not None:
    network.load_weights(config_dict["weights"])

# if "tracker" in config_dict:
#     # Load the tracker
#     tracker_configuration = config_dict["tracker"]["configuration"]
#     tracker = getattr(trackers, config_dict["tracker"]["name"])(**tracker_configuration)

#     # Load the custom model for prediction
#     model_configuration = config_dict["model"]["configuration"]
#     model_configuration["tracker"] = tracker
#     model_configuration["network"] = network
#     model = getattr(models, config_dict["model"]["name"])(**model_configuration)
#     network = model

# Load the data once before prediction
X, Y = generator.__getitem__(0)
print(config_dict["network"]["name"])

print(X["data_input"][0].shape)

if config_dict["network"]["name"] in ["DeepMotion3D_Indus"]:
    frames_per_sample = X["data_input"][0].shape[2]
elif config_dict["network"]["name"] in ["DeepMotionCLS_Indus"]:
    frames_per_sample = X["data_input"][0].shape[0] * X["data_input"][0].shape[1]
elif config_dict["network"]["name"] in ["DeepMotionCLF_Indus"]:
    frames_per_sample = X["data_input"][0].shape[0]
else:
    frames_per_sample = X["data_input"][0].shape[0]


# Estimate the prediction time
start = time.time()
outputs = []
gpu_percents = []
gpu_mem = []

for _ in tqdm(range(number_of_samples)):
    predictions = network.predict(X)

    # res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
    # mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    # print(f'mem: {mem_res.used / (1024**2)} (GiB)')
    # print(f'mem: {100 * (mem_res.used / mem_res.total):.3f}%')
    outputs.append(generator.post_process_output(predictions))
    outputs = []
    # gpu_percents.append(res.gpu)
    # gpu_mem.append(res.memory)

# mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

# final_memory = mem_res.used / (1024**2)

# network_memory = final_memory - original_memory

end = time.time()
final_time = end - start

print("Number of frames per sample: {}".format(frames_per_sample))
# print("Percentage of gpu used for calculation: {}%".format(np.mean(gpu_percents)))
# print("Memory used by the network: {} GiB".format(network_memory))

print(
    "It took {} seconds to carry out the prediction on {} batches of size {}".
    format(final_time, number_of_samples, batch_size))

samples_per_second = (number_of_samples * batch_size) / final_time

print("This makes the network able to process {} samples per second.".format(
    samples_per_second))

frames_per_seconds = samples_per_second * frames_per_sample

print(
    "Each sample contained {} frames, which makes a final result of {} frames per seconds."
    .format(frames_per_sample, frames_per_seconds))
