from os import makedirs
from os.path import join, exists
from statistics import mean

import numpy as np

from tqdm import tqdm

from sklearn.metrics import r2_score

from .helpers import extract_sample_from_matrix, extract_sample_from_matrix_v2
from .helpers import save_histogram, save_2d_distribution
from .helpers import save_hexbin, save_plot, save_error_plot, save_time_plot


from flow_estimation.generators import GeneratorMVSecond, BaseGenerator


class Evaluator(object):
    def __init__(self, round: bool = False):
        """ Base evaluator for the prediction of the QTV data.

        # Arguments:
            - round: If the Q predictions should be rounded.

        """
        self.results_metrics = None
        self.generator_data = {}
        self.round = round

    def __call__(self, generators: object, model: object,
                 output_directory: str, id: str):
        """ Function to compute the results given a model and a generator.

        # Arguments:
            - generators: A dictionnary with three keys contaning the training, validation and testing generators.
            - model: The model to evaluate against.
            - output_directory: Where the results and display should be output.
            - id: The id of the dataset being evaluated against.
        
        """
        # For each of the generator, compute the evaluation
        for key in generators:
            if key in ["training", "validation"]:
                output_key_directory = join(output_directory, key)
                continue
            else:
                output_key_directory = join(output_directory, id)

            # If the output dir already exists, we skip it as it may be training or validation already computed.
            if exists(output_key_directory) and key in [
                    "training", "validation"
            ]:
                continue

            # Create the directory that will hold the statistical analysis of the data
            statistic_directory = join(output_key_directory, "01_statistics")

            makedirs(output_key_directory)
            makedirs(statistic_directory)

            generator = generators[key]

            # Create the array to hold the data and results
            self.initialize_generator_data(generator, key)

            # For each of the batch of data, compute the statistics and evaluation
            for data_index in tqdm(range(len(generator))):
                X, Y, information = generator.__getitem_test__(data_index)
                Y = generator.post_process_output(Y)

                self.extract_information_for_statistics(
                    X["data_input"], Y, information,
                    data_index * generator.batch_size, key, generator)

                predictions = model.predict(X)

                if not isinstance(predictions, list):
                    predictions = [predictions]
                predictions = {
                    name: pred
                    for name, pred in zip(model.output_names, predictions)
                }

                predictions = generator.post_process_output(predictions)

                # Store the results for later computation of the results
                for data_key in predictions:
                    for j in range(len(predictions[data_key])):
                        if self.round:
                            self.generator_data[key]["data_output"][data_key][
                                "predicted"][data_index * generator.batch_size
                                             + j] = np.round(
                                                 predictions[data_key][j])
                        else:
                            self.generator_data[key]["data_output"][data_key][
                                "predicted"][data_index * generator.batch_size
                                             + j] = predictions[data_key][j]

            # Compute the evaluation metrics (only for the test set)
            self.compute_metrics(key)

            # Generate and save/display statistical information about the data
            self.generate_generator_statistics_display(key,
                                                       statistic_directory,
                                                       generator)

            # Generate and save/display results graph about the data
            self.generate_generator_evaluation(output_key_directory, key, id)

        return self.evaluated_results

    def initialize_generator_data(self, generator, key):
        """ Initialise the dictionnary that will hold all the information about the current generator and its evaluation results.

        # Arguments:
            - generator: The current generator, used to compute the size of the results matrices and pre-load them
            - key: The current generator key, used as dictionnary key to store the results
        """
        if key not in self.generator_data:
            self.generator_data[key] = {
                "data_input": {
                    "motion_vector_samples":
                    np.empty((len(generator) * generator.batch_size, 200, 2))
                },
                "data_output": {},
                "time_segments": {}
            }

            for label_key in generator.output_labels:
                self.generator_data[key]["data_output"][label_key] = {
                    "real":
                    np.zeros((len(generator) * generator.batch_size, 1)),
                    "predicted":
                    np.zeros((len(generator) * generator.batch_size, 1))
                }

    def extract_information_for_statistics(self, X, Y, information,
                                           start_index, generator_key,
                                           generator):
        """ Extract all the information required for the later computation of the statistics over the data.

        # Arguments:
            - X: The input data to be fed to the network, of shape (batch_size, data_shape)
            - Y: The target data matching the inputs, of shape (batch_size, data_label)
            - information: The information about the data, here the path to the data file. It is used to compute the time segment each of the points belong to.
            - start_index: Current start index of the batch, used to store the data at the correct index in the results matrices.
            - generator_key: Key to the current generator being processed
            - generator: The current generator, used for the extraction of some of the statistics as the data representation may change depending on the generator.
        """
        for i in range(len(X)):
            if isinstance(generator, GeneratorMVSecond):
                motion_vectors_sample = extract_sample_from_matrix_v2(X[i])

                self.generator_data[generator_key]["data_input"][
                    "motion_vector_samples"][start_index +
                                             i] = motion_vectors_sample
            elif isinstance(generator, BaseGenerator):
                motion_vectors_sample = extract_sample_from_matrix(X[i])

                self.generator_data[generator_key]["data_input"][
                    "motion_vector_samples"][start_index +
                                             i] = motion_vectors_sample

            # For all the possible keys, store the data
            for key in Y:
                self.generator_data[generator_key]["data_output"][key]["real"][
                    start_index + i] = Y[key][i]

            # Store the key information about the datapoint
            splitted_path = information[i].split("/")
            segment_key = splitted_path[-4].replace(
                "_", "-") + ":" + splitted_path[-3].replace("_", ":")
            if not segment_key in self.generator_data[generator_key][
                    "time_segments"]:
                self.generator_data[generator_key]["time_segments"][
                    segment_key] = [start_index + i]
            else:
                self.generator_data[generator_key]["time_segments"][
                    segment_key].append(start_index + i)

    def generate_generator_statistics_display(self, generator_key,
                                              output_directory, generator):
        """ Generate the statistical analysis for the target generator.

        # Arguments:
            - generator_key: The key of the target generator for the analysis.
            - output_directory: The directory where the results should be saved.
        """
        # Get all the pre-computed information for the given generator key
        generator_data = self.generator_data[generator_key]

        time_segments = generator_data["time_segments"]

        for key in generator_data["data_output"]:
            if key == "T":
                plot_range = range(0, 100, 1)
            else:
                plot_range = range(0, 30, 1)

            data = np.array(generator_data["data_output"][key]["real"])

            # Generate dummy indexes to select all the data points
            global_data_indexes = {key: [i for i in range(len(data))]}

            key_output_directory = join(output_directory, key)
            makedirs(key_output_directory)

            # First for each of the time segments, save the histogram of the values
            save_histogram(data,
                           time_segments,
                           output_directory,
                           "{}_time_segments".format(key),
                           data_range=plot_range)

            # Then plot the global histogram
            save_histogram(data,
                           global_data_indexes,
                           output_directory,
                           "{}".format(key),
                           data_range=plot_range)

        motion_vector_samples = generator_data["data_input"][
            "motion_vector_samples"]
        global_mv_indexes = {
            "Global Distribution":
            [i for i in range(len(motion_vector_samples))]
        }
        # Plot the motion vector distribution, first for every time segment
        save_2d_distribution(motion_vector_samples, time_segments,
                             output_directory, generator_key,
                             generator.convertion)

        # Then plot the global motion vector distribution
        save_2d_distribution(motion_vector_samples, global_mv_indexes,
                             output_directory, generator_key,
                             generator.convertion)

    def compute_metrics(self, key_generator):
        """ Compute the results for the test generator.

        # Argument:
            - key_generator: The key of the current generator being processed.
        """
        if key_generator != "test":
            return

        results = {}

        generator = self.generator_data[key_generator]
        time_segments = generator["time_segments"]

        for key in generator["data_output"]:
            y_true = generator["data_output"][key]["real"]
            y_pred = generator["data_output"][key]["predicted"]

            # These coefficients are global
            r = np.corrcoef(y_true.T, y_pred.T)

            abs_errors = np.abs(y_pred - y_true)

            mae = np.mean(abs_errors)

            # This coefficient is not
            r2_global = 0
            r2_dict = {}
            for pred_key in time_segments:
                r2_dict[pred_key] = r2_score(y_true[time_segments[pred_key]],
                                             y_pred[time_segments[pred_key]])

                r2_global += r2_dict[pred_key]

            r2_global /= len(time_segments.keys())

            results[key] = {"MAE": mae, "R": r[0][1], "R2": r2_global}

            self.results_metrics = sorted(results[key].keys())

        # self.evaluated_results = results
        self.evaluated_results = [mae, r[0][1], r2_global]

    def generate_generator_evaluation(self, output_directory, generator_key,
                                      experiment_id):
        """ Generate various statistical graph about the predictions of the current generator.

        # Arguments:
            - output_directory: The directory where the graphs will be output.
            - generator_key: The key of the current generator being processed.
            - experiment_id: The id of the current experiment, used for the naming of some of the graphs.

        # TODO:
            Merge this function with the one above computing the results.

        """
        generator_data = self.generator_data[generator_key]
        key_dict = {
            key: "{:02d}_{}".format(i + 2, key)
            for i, key in enumerate(
                sorted(generator_data["data_output"].keys()))
        }

        # Plot the graph for each of the predicted outputs
        for key in generator_data["data_output"]:

            y_true = np.array(generator_data["data_output"][key]["real"])
            y_pred = np.array(generator_data["data_output"][key]["predicted"])
            time_segments = generator_data["time_segments"]

            # Create the output directory
            output_directory_key = join(output_directory, key_dict[key])
            makedirs(output_directory_key, exist_ok=True)

            # First save the hexbin map of the prediction vs real data
            save_hexbin(y_true, y_pred, output_directory_key, key,
                        experiment_id)

            # Then the "normal" map of the prediction vs real data
            save_plot(y_true, y_pred, output_directory_key, key, experiment_id)

            # Then plot the data over time
            save_time_plot(y_true, y_pred, time_segments, output_directory,
                           key, experiment_id)

            # Then plot the error graph
            save_error_plot(y_true, y_pred, output_directory_key, key,
                            experiment_id)
