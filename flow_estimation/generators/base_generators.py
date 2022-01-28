from os import environ, listdir
from os.path import join, isfile

from typing import Dict

from tqdm import tqdm

import h5py

import numpy as np

from tensorflow.keras.utils import Sequence


class BaseGenerator(Sequence):
    def __init__(self,
                 set_file: str,
                 batch_size: int = 16,
                 shuffle: bool = True,
                 extract_Q: bool = True,
                 extract_T: bool = False,
                 normalize: Dict = None,
                 convertion: str = None,
                 disable: bool = True,
                 seed: int = 864,
                 set_type: str = None):
        """ Base class for the generators. Only provides Motion Vectors.

        # Arguments:
            - set_file: The file containing the directories to be looked in for the generation of the data.
            - batch_size: The size of the batches to be returned by the generator, default is 16.
            - shuffle: If the data should be shuffled, default is True.
            - extract_Q: If the Q values should be returned as Y.
            - extract_T: If the T values should be returned as Y.
            - normalize: The normalization dictionnary. See the `Details` section.
            - convertion: If the motion vectors should be converted to another format (polar coordinate or module of the polar coordinates only).
            - disable: If the display of the progress bars should be disabled or not
            - seed: To be used with `set_type`, fix the randomization to allow the same split when using training or validation
            - set_type: The set to generate, if None all the data is used, if `training`, 80 percent of the data is used, if `validation`, 20 percent of the data is used. The datapoint are shuffled based on seed before being associated with a set.

        # Details:
            The normalization parameters takes the form of the following dictionnary (default values used as example):
            ```json
                {
                    "Q": {
                            "mean": 0,
                            "std": 1
                         },
                    "T": {
                            "mean": 0,
                            "std": 1
                         }
                }
            ```
        """
        ## Check for the runtime errors
        if not (extract_Q or extract_T):
            raise RuntimeError(
                "At least one of extract_Q, extract_T should be True.")

        if not (convertion is None or convertion in ["polar", "polar_module"]):
            raise RuntimeError(
                "The convertion parameter should be one of [None, 'polar', 'polar_module'], not {}".format(convertion))

        ## Set the parameters
        self.__extract_Q = extract_Q
        self.__extract_T = extract_T

        self.convertion = convertion
        self.output_labels = set([])

        if self.extract_Q:
            self.output_labels.add("Q")
        if self.extract_T:
            self.output_labels.add("T")

        self.__batch_size = batch_size
        self.__shuffle = shuffle
        self.disable = disable

        if normalize is None:
            self.normalize = {
                "T": {
                    "mean": 0,
                    "std": 1
                },
                "Q": {
                    "mean": 0,
                    "std": 1
                }
            }
        else:
            self.normalize = normalize

        # Variable that will hold all the data files
        self.files = []

        # Set the parameter for the set to select
        self.seed = seed
        self.set_type = set_type

        ## Extract the datapoints
        set_file = join(environ["DATASET_PATH"], set_file)

        # Look for all the data points in the folders provided in the set_file
        with open(set_file, "r") as file:
            main_directories = [
                join(environ["DATASET_PATH"], line.strip())
                for line in file.readlines()
            ]

        for directory in tqdm(main_directories,
                              desc="Extracting the files from the set_file.",
                              disable=self.disable):
            data_path = join(directory, "compressed_data")
            self.files += sorted([
                join(data_path, file) for file in listdir(data_path)
                if isfile(join(data_path, file)) and not "residual" in file
            ])

        # Checking the files for -1 values and removing them
        for i in tqdm(range(len(self.files) - 1, -1, -1), disable=self.disable):
            with h5py.File(self.files[i], "r") as h5py_file:
                Qs = h5py_file["motion_vectors"].attrs["q"]
                if not isinstance(Qs, np.int64) and Qs[0] == -1:
                    self.files.pop(i)

        ## Create the index for the generation of the batches
        self.index = np.arange(len(self.files))

        # If a set is selected, split the dataset and reset the indexes
        if set_type is not None:
            np.random.seed(self.seed)
            np.random.shuffle(self.index)
            n = int(len(self.index) * 0.8)

            if set_type is "training":
                self.index = self.index[:n]
            elif set_type == "validation":
                self.index = self.index[n:]

        # defining the parameters for the inner functions
        self.number_of_batch = len(self.index) // batch_size

        self.on_epoch_end()

    def __len__(self):
        return self.number_of_batch

    def on_epoch_end(self):
        if self.__shuffle:
            np.random.shuffle(self.index)

    def __getitem_test__(self, index):
        """ Returns the items at test time. It will also return the information available on the points generated for more analysis.

        # Arguments:
            - index: The index of the batch to return

        # Return:
            Three outputs, X data, Y data and the information related to each data point. For now the information is just the path of the data file (this is used for evaluation).

        """
        X, Y = self.__getitem__(index)

        index = index % self.number_of_batch

        indexes = self.index[index * self.batch_size:(index + 1) *
                             self.batch_size]

        information = [self.files[ind] for ind in indexes]

        return X, Y, information

    def __getitem__(self, index):
        """ Return a the batch at the given index.

        # Arguments:
            - index: The index of the batch to return

        # Return:
            A complete batch of data, two outputs, the X data and the Y data.
        """
        raise NotImplementedError()

    def post_process_output(self, results):
        """ Post-process the output in a usable format."""
        raise NotImplementedError(
            "The function post_process_output is not yet implemented.")

    # Setter and getters to modify the parameters of a given generator on the fly
    @property
    def extract_Q(self):
        return self.__extract_Q

    @extract_Q.setter
    def extract_Q(self, extract_Q):
        if not extract_Q:
            self.output_labels.remove("Q")
        else:
            self.output_labels.add("Q")
        self.__extract_Q = extract_Q

    @property
    def extract_T(self):
        return self.__extract_T

    @extract_T.setter
    def extract_T(self, extract_T):
        if not extract_T:
            self.output_labels.remove("T")
        else:
            self.output_labels.add("T")
        self.__extract_T = extract_T

    @property
    def shuffle(self):
        return self.__shuffle

    @shuffle.setter
    def shuffle(self, shuffle):
        self.__shuffle = shuffle
        # Reset the indices depending on the new value of the shuffle
        if shuffle:
            # For now, on_epoch_end only shuffles the data
            self.on_epoch_end()
        else:
            # Re-order the indices
            self.index = np.sort(self.index)

    @property
    def batch_size(self):
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        self.__batch_size = batch_size
        self.number_of_batch = len(self.index) // batch_size
