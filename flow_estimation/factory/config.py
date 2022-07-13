import json

from typing import Dict

from os import environ
from os.path import join

from copy import deepcopy

from tensorflow.keras import optimizers, callbacks
from tensorflow.keras import losses as keras_losses
from tensorflow.keras import backend as K
from tensorflow.keras import backend as K_2

from flow_estimation import networks, generators, evaluators, displayers

possible_losses = [keras_losses]


class TrainingConfiguration(object):
    def __init__(self, config_dict, disable=True):
        """Configuration class to load objects required for training.

        # Arguments:
            - config_dict: The dictionnary extracted by a factory object.
            - disable: Whether the loading display for the various objects to be generated should be enabled or not.
        """
        self.config_dict = config_dict

        # Setting the fitting parameters
        self.fit_parameters = deepcopy(config_dict["fit_parameters"])

        # Setting the name of the experiment (used for later storage)
        self.experiment_name = config_dict["experiment_name"]
        self.sub_experiment_name = config_dict["sub_experiment_name"]
        self.weights = config_dict["weights"]

        # Creating the network
        network_configuration = config_dict["network"]["configuration"]

        self.network = getattr(
            networks, config_dict["network"]["name"])(**network_configuration)

        if config_dict["weights"] is not None:
            self.network.load_weights(config_dict["weights"])

        # Create the optimizer
        optimizer_configuration = config_dict["optimizer"]["configuration"]
        self.optimizer = getattr(
            optimizers,
            config_dict["optimizer"]["name"])(**optimizer_configuration)

        # Loading the loss and metrics
        loss_configuration = config_dict["loss"]["configuration"]

        for loss_package in possible_losses:
            loss = getattr(loss_package, config_dict["loss"]["name"], None)
            if loss is not None:
                break
        else:
            raise RuntimeError("Loss {} not found.".format(
                config_dict["loss"]["name"]))

        self.loss = loss(**loss_configuration)

        self.metrics = config_dict["metrics"]

        # Generating the data (generator or full load)
        generator_configuration = config_dict["generator"]["train"][
            "configuration"]

        self.fit_parameters["generator"] = getattr(
            generators, config_dict["generator"]["train"]["name"])(
                **generator_configuration)

        # Loading the validation generator if any
        if not config_dict["generator"]["validation"] is None:
            generator_configuration = config_dict["generator"][
                "validation"]["configuration"]

            self.fit_parameters["validation_data"] = getattr(
                generators,
                config_dict["generator"]["validation"]["name"])(
                    **generator_configuration)

        # If steps per epochs is not define, use the len of the main generator
        if "steps_per_epoch" not in self.fit_parameters:
            self.fit_parameters["steps_per_epoch"] = len(
                self.fit_parameters["generator"])

        # If validation_steps is not define, use the len of the validation generator
        if ("validation_steps"
                not in self.fit_parameters) and ("validation_data" in self.fit_parameters):
            self.fit_parameters["validation_steps"] = len(self.fit_parameters["validation_data"])

        # Load the static callbacks and keep the runtime for later
        self.runtime_callbacks_dict = config_dict["callbacks"]["runtime"]

        self.callbacks = []
        static_callbacks = config_dict["callbacks"]["static"]

        for callback in static_callbacks:
            callback_configuration = callback["configuration"]
            self.callbacks.append(
                getattr(callbacks, callback["name"])(**callback_configuration))

        # Setting the fit parameters with the callbacks
        self.fit_parameters["callbacks"] = self.callbacks

    def clear(self):
        """ Cleanup function, only clears the tensorflow graph."""
        K.clear_session()
        K_2.clear_session()

    def prepare_runtime_callbacks(self, directories_dir):
        log_dir = directories_dir["log_dir"]
        checkpoints_dir = directories_dir["checkpoints_dir"]

        for callback in self.runtime_callbacks_dict:
            if callback["name"] == "ModelCheckpoint":
                self.callbacks.append(
                    callbacks.ModelCheckpoint(filepath=join(
                        checkpoints_dir, "best_weights.h5"),
                                              save_best_only=False))
            elif callback["name"] == "TensorBoard":
                callback_configuration = callback["configuration"]
                callback_configuration["log_dir"] = log_dir
                self.callbacks.append(
                    callbacks.TensorBoard(**callback_configuration))
            else:
                raise RuntimeError("Callback name: {} is not supported.".format(
                    callback["name"]))

    def save(self, save_path):
        """ Function to save the current configuration as a json file.

        # Arguments:
            - save_path: Where to save the json file.
        """
        save_file_path = join(save_path, "config.json")

        with open(save_file_path, 'w') as fp:
            json.dump(self.config_dict, fp)


class TestingConfiguration(object):
    def __init__(self, config_dict: Dict):
        """ Configuration class to load objects required for testing.

        # Arguments:
            - config_dict: The dictionnary extracted by a factory object.
        """
        self.config_dict = config_dict

        # Variable to hold the parameters for the evaluator
        self.evaluation_parameters = {}

        # Setting the id parameters (global to all models)
        self.experiment_name = config_dict["experiment_name"]
        self.dataset_name = config_dict["dataset_name"]
        self.dataset_id = config_dict["dataset_id"]
        
        # Loading the evaluator (global to all models)
        evaluator_configuration = config_dict["evaluator"]["configuration"]

        self.evaluator = getattr(
            evaluators,
            config_dict["evaluator"]["name"])(**evaluator_configuration)

        # Loading the parameter that may differ between experiments
        self.load_base_configuration(config_dict)
    
    def load_base_configuration(self, config_dict):
        """ Base loading function for the evaluation of the tested methods.
            This function is called when no specific modification are needed on the configuration files for the evaluation.
            This function will:
                - Load the network to be evaluated
                - Load the training generator (used for checkup of the training)
                - Load the validation generator if any (also used for checkup of the training)
                - Load the testing generator
        
        # Argument:
            - config_dict: The dictionnary containing all the information to be loaded.
        """
        # We load the network and the weights
        network_configuration = config_dict["network"]["configuration"]

        network = getattr(
            networks, config_dict["network"]["name"])(**network_configuration)
        network.load_weights(config_dict["weights"])

        # Loading the training generator as we will use it for training verification
        generator_configuration = config_dict["generator"]["train"][
            "configuration"]

        generator_configuration["shuffle"] = False
        generator_configuration["batch_size"] = 8

        train_generator = getattr(
            generators, config_dict["generator"]["train"]["name"])(
                **generator_configuration)

        # Loading the validation generator if any
        if not config_dict["generator"]["validation"] is None:
            generator_configuration = config_dict["generator"][
                "validation"]["configuration"]

            generator_configuration["shuffle"] = False
            generator_configuration["batch_size"] = 8

            validation_generator = getattr(
                generators,
                config_dict["generator"]["validation"]["name"])(
                    **generator_configuration)
        else:
            validation_generator = None

        # Loading the test generator
        generator_configuration = config_dict["generator"]["test"][
            "configuration"]

        generator_configuration["shuffle"] = False
        generator_configuration["batch_size"] = 8
        generator_configuration["set_file"] = join(
            environ["DATASET_PATH"], generator_configuration["set_file"])

        test_generator = getattr(
            generators, config_dict["generator"]["test"]["name"])(
                **generator_configuration)

        # Prepare the dictionnary with all the parameters to be used in the evaluation 
        self.evaluation_parameters["model"] = network
        self.evaluation_parameters["id"] = config_dict["dataset_id"]
        self.evaluation_parameters["generators"] = {}
        self.evaluation_parameters["generators"]["training"] = train_generator
        self.evaluation_parameters["generators"]["test"] = test_generator
        
        # Maybe load the validation generator
        if validation_generator is not None:
            self.evaluation_parameters["generators"]["validation"] = validation_generator

    def clear(self):
        """ Cleanup function, mostly to handle keras stuff. """
        K.clear_session()
        K_2.clear_session()

    def prepare_runtime_stuff(self, experiment_output_directory: str):
        """ Function to load the runtime variables.

        # Arguments:
            - experiment_output_path: The path were the data created by the evaluator will be output.
        """
        self.evaluation_parameters[
            "output_directory"] = experiment_output_directory

class DisplayConfiguration(object):
    def __init__(self, config_dict : Dict, disable: bool=True):
        """ Configuration class to load objects required for display.

        # Arguments:
            - config_dict: The dictionnary extracted by a factory object.
            - disable: Whether the loading display for the various objects to be generated should be enabled or not.
        """
        # Creating the displayer
        network_configuration = config_dict["displayer"]["configuration"]

        self.displayer = getattr(
            displayers, config_dict["displayer"]["name"])(**network_configuration)

        # Creating the network
        network_configuration = config_dict["network"]["configuration"]
        network_configuration["mode"] = "display"

        self.network = getattr(
            networks, config_dict["network"]["name"])(**network_configuration)

        # Creating the generator
        generator_configuration = config_dict["generator"][
            "test"]["configuration"]

        self.test_generator = getattr(
            generators,
            config_dict["generator"]["test"]["name"])(
                **generator_configuration)
