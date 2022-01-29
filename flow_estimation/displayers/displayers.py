import cv2

import numpy as np

import os


class Displayer(object):
    def __init__(self):
        """ Displayer for any of the networks based on direct estimation of the QT values from the motion vectors.
            The display will be done using the RGB video and the resized motion vectors side by side.
        """

    def display_with_gt(self,
                        network: object,
                        generator: object,
                        num_datapoints: int = 4,
                        output_dir: str = None,
                        shuffle: bool = True):
        """ Function to display the prediction with the groundtruth.

        # Arguments:
            - network: The network to be used for prediction.
            - generator: The generator to get the data from.
            - num_datapoints: The number of datapoints to be displayed
            - output_dir: If specified, where to output the predictions
        """
        self.display(network, generator, num_datapoints, output_dir, shuffle, True)

    def display(self,
                network: object,
                generator: object,
                num_datapoints: int = 4,
                output_dir: str = None,
                shuffle: bool = True,
                groundtruth=False):
        """ Function to display the prediction of a network.

        # Arguments:
            - network: The network to be used for prediction.
            - generator: The generator to get the data from.
            - num_datapoints: The number of datapoints to be displayed
            - output_dir: If specified, where to output the predictions
        """
        # Set the generator with the provided parameters
        generator.shuffle = shuffle
        generator.batch_size = 1

        # For the number of datapoints to display
        for point_idx in range(num_datapoints):
            # Get the data to display
            data_batch, output_batch = generator.__getitem__(point_idx)

            output_batch = generator.post_process_output(output_batch)

            # Extract the required information for display
            rgb_image = data_batch["rgb_path"][0]

            # Predict and convert to dictionnary
            network_output = network.predict(data_batch)
            network_output = {name: pred for name, pred in zip(network.output_names, network_output)}

            network_output = generator.post_process_output(network_output)

            # Open the original rgb file
            cap = cv2.VideoCapture(rgb_image)

            if (cap.isOpened() == False):
                raise RuntimeError("Error opening video stream or file")

            # Get the size of the input rgb frames and number of frames
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # If the output is to be stored 
            if output_dir is not None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                out = cv2.VideoWriter(
                    os.path.join(output_dir, "{:02d}.mp4".format(point_idx + 1)),
                    fourcc, 25, (width * 2, height), True)

            # Display the results
            for _ in range(num_frames):

                ret, frame = cap.read()
                
                # Set the frame for display
                final_frame = np.ones((height, width * 2, 3), dtype=np.uint8) * 255
                final_frame[:, :width] = frame

                # Write the predictions on screen
                color = (0, 0, 0)

                cv2.putText(final_frame,
                            "Prediction: {:.2f}".format(network_output["Q"][0]),
                            (1 * width + 25, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 1)
                
                if groundtruth:
                    cv2.putText(final_frame,
                            "Ground Truth: {:.2f}".format(output_batch["Q"][0]),
                            (1 * width + 25, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 1)


                # Display the frame
                cv2.imshow("Datapoint {:02d}".format(point_idx + 1),
                        final_frame)

                if output_dir is not None:
                    out.write(final_frame)

                # Press Q on keyboard to  exit
                if cv2.waitKey(40) & 0xFF == ord('q'):
                    break
            # Close all
            cv2.destroyWindow("Datapoint {:02d}".format(point_idx + 1))
            cap.release()
            if output_dir is not None:
                out.release()
