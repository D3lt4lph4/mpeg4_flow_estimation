import json
import collections.abc

from copy import deepcopy


def update(a, b):
    """ Updates the values in dictionnary a with the values in dictionnary b.

    # Arguments:
        - a: The dictionnary to be updated.
        - b: The dictionnary containing the values to update.
    
    # Returns:
        The dictionnary a. No copy is performed, the returned value is the same pointer as the argument one.
    """
    for key, value in b.items():
        if isinstance(value, collections.abc.Mapping):
            a[key] = update(a.get(key, {}), value)
        else:
            a[key] = value

    return a


class TrainingConfigurationFactory(object):

    def __init__(self, config_file, repeat):
        """ Factory to generate the configuration dictionaries for a given experiment.

        # Arguments:
            - config_file: The configuration file that will be used to generate the configuration dictionnary for every sub-experiment.
        """
        # Read the configuration file
        with open(config_file, "r") as json_file:
            dictionnary = json.load(json_file)

        # Load the parameters common to every sub-experiment
        self.common_parameters = dictionnary["common"]

        # Load the different datasets to use for the trainings
        self.dataset_iterator = dictionnary["dataset"]

        # Load the parameters to iterate upon
        self.experiment_parameters = dictionnary["experiment_variables"]

        self.repeat = repeat

    def __iter__(self):
        # We iterate over the experiment parameters
        for i, experiment in enumerate(self.experiment_parameters):
            # Set the sub-experiment name
            sub_experiment_name = experiment["sub_experiment_name"]
            # Iterate over each of the datasets
            for j, dataset in enumerate(self.dataset_iterator):
                for k in range(self.repeat):
                    # Copy the parameters common to all the experiments
                    common_parameters = deepcopy(dict(self.common_parameters))

                    # Add a number to the sub-experiment for storing the outputs
                    experiment["sub_experiment_name"] = "{:02d}_{}".format(
                        i * len(self.dataset_iterator) + j * self.repeat + k + 1, sub_experiment_name)

                    # Update the common parameters with the current experiment parameters
                    parameters = update(common_parameters,
                                        experiment)

                    # Update the parameters with the current dataset
                    parameters = update(parameters, dataset)

                    # Set the dataset tag, this will be used to display the results
                    set_file = parameters["generator"]["train"]["configuration"]["set_file"]
                    if isinstance(set_file, list):
                        dataset_tag_name = set_file[1].split("/")[-2].split("_")[-1]
                    else:
                        dataset_tag_name = set_file.split("/")[-2].split("_")[-1]


                    # Add the id of the training set to the sub-experiment name
                    parameters["sub_experiment_name"] = parameters["sub_experiment_name"] + \
                        "_{}".format(dataset_tag_name)

                    # Yield the new configuration
                    yield deepcopy(parameters)
