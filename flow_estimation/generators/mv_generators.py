import numpy as np

import h5py

from .base_generators import BaseGenerator


class GeneratorMVSecond(BaseGenerator):
    def __init__(self,
                 set_file: str,
                 batch_size: int = 16,
                 shuffle: bool = True,
                 extract_Q: bool = True,
                 extract_T: bool = False,
                 normalize=None,
                 convertion: str = None,
                 disable: bool = True,
                 seed: int = 864,
                 set_type: str = None):
        """ Generate batch of motion vector data.

        # Arguments:
            See BaseGenerator class.
        """
        super(GeneratorMVSecond, self).__init__(set_file,
                                                   batch_size,
                                                   shuffle,
                                                   extract_Q,
                                                   extract_T,
                                                   normalize,
                                                   convertion,
                                                   disable,
                                                   seed,
                                                   set_type)
        if not disable:
            print("Done init the generator.")

    def __getitem__(self, index):
        """ Return a the batch at the given index.

        # Arguments:
            - index: The index of the batch to return

        # Return:
            A complete batch of data.
        """
        # If the index is above the number of batch, modulo
        index = index % self.number_of_batch

        indexes = self.index[index * self.batch_size:(index + 1) *
                             self.batch_size]

        data_files = [self.files[ind] for ind in indexes]

        X = []
        Y = []

        for data_file in data_files:

            with h5py.File(data_file, "r") as h5py_file:

                # Extract and process the matrices of motion vectors
                x = h5py_file["motion_vectors"][:]

                w, h, _, _ = x.shape

                if self.convertion in ["polar", "polar_module"]:
                    x = x.astype(np.float32)
                    x[:, :, :, 0], x[:, :, :, 1] = np.sqrt(np.power(x[:, :, :, 0], 2) + np.power(x[:, :, :, 1], 2)), np.arctan2(
                        x[:, :, :, 1], x[:, :, :, 0], out=np.zeros_like(x[:, :, :, 1]), where=x[:, :, :, 0] != 0)

		# Reshape the data as it is stored as one big matrix (w,h,n_frames)
                if self.convertion == "polar_module":
                    x = x[:, :, :, 0]
                    x = x.reshape((w, h, -1, 25)).transpose(
                        (2, 3, 0, 1))
                else:
                    x = x.reshape((w, h, -1, 25, 2)).transpose(
                        (2, 3, 0, 1, 4))

                # Extract the QT information
                Qs = h5py_file["motion_vectors"].attrs["q"]
                Ts = h5py_file["motion_vectors"].attrs["t"]

		# Normalize the data if required
                if self.extract_Q and self.extract_T:
                    y = np.zeros(2)
                    y[0] = (np.mean(Qs) - self.normalize["Q"]["mean"]) / self.normalize["Q"]["std"]
                    y[1] = (np.mean(Ts) - self.normalize["T"]["mean"]) / self.normalize["T"]["std"]
                else:
                    y = np.zeros(1)
                    if self.extract_Q:
                        y[0] = (np.mean(Qs) - self.normalize["Q"]["mean"]) / self.normalize["Q"]["std"]
                    else:
                        y[0] = (np.mean(Ts) - self.normalize["T"]["mean"]) / self.normalize["T"]["std"]

                X.append(x)
                Y.append(y)

        return {"data_input": np.array(X)},  {"predictions": np.array(Y)}

    def post_process_output(self, results):
        """ Apply any post-processing required to read the data generated by the generator and prediction network (de-normalize, ...).

        # Arguments:
            - results: Either the output data generated by the generator or the output generated by a prediction network.
        """
        mean_Q = self.normalize["Q"]["mean"]
        std_Q = self.normalize["Q"]["std"]

        mean_T = self.normalize["T"]["mean"]
        std_T = self.normalize["T"]["std"]

        if isinstance(results, dict):
            raw_results = results["predictions"]
        else:
            raw_results = results

        if self.extract_T and self.extract_Q:
            return {"Q": (raw_results[:, 0] * std_Q) + mean_Q, "T": (raw_results[:, 1] * std_T) + mean_T}
        elif self.extract_Q:
            return {"Q": (raw_results * std_Q) + mean_Q}
        else:
            return {"T": (raw_results * std_T) + mean_T}
