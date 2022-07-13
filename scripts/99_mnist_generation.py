import argparse
from typing import Tuple, Dict, List

from os.path import basename

from multiprocessing import Pool

import h5py

import warnings
warnings.filterwarnings("ignore")

from tensorflow import keras

import json

import numpy as np

import cv2
import random

import os
from os import makedirs
from os.path import join

from tqdm import tqdm

# from mv_utils import load_gops_size
import mpeg42compressed.numpy.extract_mvs as extract_mvs

import subprocess

from time import sleep

## GLOBAL PARAMETERS
N_SECONDS = 20
FPS = 25.0
IMAGE_DIMS = (200, 200)


## FUNCTIONS DEFINITION
def convert(img: object,
            img_min: int = None,
            img_max: int = None,
            target_type_min: int = 0,
            target_type_max: int = 255,
            target_type: object = None):
    """ Converts an image to a given range of values.
    
        # Arguments:
            - img: The image to convert, a numpy array.
            - img_min: The min possible value of the image, if None, the min of the image is used.
            - img_max: The max possible value of the image, if None, the max of the image is used.
            - target_type_min: Target min value once converted.
            - target_max_type: Target max value once converted.
            - target_type: The numpy type to convert to.
        
        # Returns
            The converted image. The converted image is not a pointer to the original image.
    """
    if img_min is None:
        img_min = img.min()

    if img_max is None:
        img_max = img.max()

    if target_type is None:
        target_type = np.uint8

    # Scaling factor
    a = (target_type_max - target_type_min) / (img_max - img_min)
    # Centering factor
    b = target_type_max - a * img_max

    # Put the image in the desired range and convert to the required type
    new_img = (a * img + b).astype(target_type)

    return new_img


class Track(object):
    def __init__(self,
                 object_image: object,
                 start_position: Tuple[int, int] = (-40, 100),
                 end_position: Tuple[int, int] = (240, 100),
                 speed: int = 150):
        """ Base object representing a track. Contains the image of the object to display, the current position of the object and the noisy operations to apply to the object image.

        # Arguments:
            - object_image: The numpy array of the image that represent the object.
            - start_position: Tuple of int containing the starting position of the object.
            - end_position: Tuple of int contaning the end position of the object.
            - speed: The displacement speed of the object, the number of frame the object will take to go from start_position to end_position
        """
        self.image = object_image

        self.object_position = 0
        self.first_position = None

        # Compute the intermediate positions and store them
        intermediate_x_positions = np.linspace(start_position[0],
                                               end_position[0],
                                               num=speed).astype(np.int16)
        intermediate_y_positions = np.linspace(start_position[1],
                                               end_position[1],
                                               num=speed).astype(np.int16)

        self.positions = np.column_stack(
            (intermediate_x_positions, intermediate_y_positions))

        self.kernel = np.ones((2, 2), np.uint8)
    
    def distance_to_first_position(self):
        if self.first_position is None:
            return 99999
        return np.linalg.norm(self.positions[self.object_position] - self.first_position)

    def set_first_position(self):
        if self.first_position is None:
            self.first_position = [self.positions[self.object_position]]

    def get_position(self):
        """ Get the track current position.

        # Return:
            A tuple, the position of the track.
        """
        position = self.positions[self.object_position]

        return position

    def update_position(self):
        """ Update the position of the object to the next position."""
        self.object_position += 1

    def is_done(self):
        """ Tells if the object has reach its end position.

        # Return:
            True if the object has reach its end position, else False
        """
        return self.object_position >= len(self.positions) - 1

    def get_object_image(self):
        """ Returns the object image and applies the corresponding noise operations if required.

            # Returns
                The object image after noise operations.
        """
        # Generate the noise on the image
        rand_num = np.random.random()
        if rand_num < 0.3:
            return cv2.erode(self.image, self.kernel, iterations=1)
        elif rand_num < 0.6:
            return cv2.dilate(self.image, self.kernel, iterations=1)
        else:
            return self.image


def update_tracks(current_object_number,
                  generation_parameters: Dict,
                  tracks: List[object],
                  current_frame: int,
                  last_updated:int,
                  update_ratio: float = 0.01,
                  max_objects: int = 20,
                  min_before_update: int = 10):
    """ Update the currents tracks.

    First adds a new track at random. Then remove the tracks that went off the screen.

    # Arguments:
        generation_parameters: Dictionnary containing all the parameters to generate a track
        tracks: The list of all the tracks
        update_ration: The percentage of time a track will be added, between [0, 1] 
    """
    n_added = 0
    last_updated += 1

    if (current_object_number >= max_objects or current_frame > (
            int(FPS * N_SECONDS) - 25)) or last_updated < min_before_update:
        pass
    elif np.random.random() < update_ratio:
        # Select a random image from the possible set of images
        image_idx = np.random.randint(0, len(generation_parameters["images"]))

        new_image = generation_parameters["images"][image_idx]

        if "scale_factor" in generation_parameters:
            scale_factor = generation_parameters["scale_factor"]
            width = int(new_image.shape[1] * scale_factor)
            height = int(new_image.shape[0] * scale_factor)
            dim = (width, height)
    
            # resize image
            new_image = cv2.resize(new_image, dim, interpolation = cv2.INTER_AREA)

        starting_position = generation_parameters["starting_position"]
        ending_position = generation_parameters["ending_position"]
        speed = generation_parameters["speed"]

        x_start = np.random.randint(*starting_position[0])
        x_end = np.random.randint(*ending_position[0])
        y_start = np.random.randint(*starting_position[1])
        y_end = np.random.randint(*ending_position[1])
        speed = np.random.randint(*speed)

        tracks.append(
            Track(new_image, (x_start, y_start), (x_end, y_end), speed))

        n_added = 1
        last_updated = 0


    return [track for track in tracks if not track.is_done()], n_added, last_updated


def update_frame(frame: object, track: object):
    """ Update the current main frame with the track.

    # Arguments:
        - frame: The frame to be updated (numpy array)
        - track: Track object to be added to the frame. 
    """
    track_image = track.get_object_image()
    track_position = track.get_position()
    track.update_position()

    h, w, _ = frame.shape
    h_o, w_o = track_image.shape

    # Check if the object is on the image at all and exit if not
    if track_position[0] + w_o <= 0 or track_position[
            0] > w or track_position[1] + h_o <= 0 or track_position[1] > h:
        return False, False

    # Compute the position of the track image on the frame
    x_min = max(track_position[0] - w_o // 2, 0)
    x_max = min(track_position[0] + w_o - (w_o // 2), w)

    y_min = max(track_position[1] - h_o // 2, 0)
    y_max = min(track_position[1] + h_o - (h_o // 2), h)

    # If the image is on the frame
    if (y_max - y_min) > 0 and (x_max - x_min) > 0:
        # Compute the portion of the object on the image
        x_min_o = 0 if track_position[0] - w_o // 2 >= 0 else -(
            track_position[0] - w_o // 2)
        x_max_o = w_o if track_position[0] + w_o - (w_o // 2) < w else w - (
            track_position[0] + (w_o // 2))

        y_min_o = 0 if track_position[1] - h_o // 2 >= 0 else -(
            track_position[1] - h_o // 2)
        y_max_o = h_o if track_position[1] + h_o - (h_o // 2) < h else h - (
            track_position[1] + (h_o // 2))

        # add the object to the frame
        frame[y_min:y_max, x_min:x_max, 0] += track_image[y_min_o:y_max_o,
                                                          x_min_o:x_max_o]
        frame[y_min:y_max, x_min:x_max, 1] += track_image[y_min_o:y_max_o,
                                                          x_min_o:x_max_o]
        frame[y_min:y_max, x_min:x_max, 2] += track_image[y_min_o:y_max_o,
                                                          x_min_o:x_max_o]
    
    is_on_screen = (y_max - y_min) > int(h_o * 0.7) and (x_max - x_min) > int(
        w_o * 0.7)
    
    if is_on_screen:
        track.set_first_position()

    return is_on_screen, track.distance_to_first_position() <= w_o


def generate_video(filename: str,
                   generation_parameters: Dict,
                   display: bool = False):
    """ Generate a data mp4 video.
        
    # Arguments:
        - filename: The name of the file to be generated.
        - generation_parameters: The parameters to be used for track generation.
        - display: If the generated images should be displayed (slows down a lot)

    """
    max_objects = [generation_parameters["lanes"][lane_index]["total_max_objects"]for lane_index in range(len(generation_parameters["lanes"]))]
    last_updated = [999 for lane_index in range(len(generation_parameters["lanes"]))]
    update_ratio = [random.choice(generation_parameters["lanes"][lane_index]["update_ratio"]) for lane_index in range(len(generation_parameters["lanes"]))]

    # Open video where the data will be saved
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, FPS, IMAGE_DIMS)

    main_data = {"q": 0, "objects_per_frames": [], "q_per_lane": [0 for _ in range(len(generation_parameters["lanes"]))]}

    tracks = []

    total_T = 0

    # For the duration of the video, generate frames
    for i in range(int(FPS * N_SECONDS)):

        # Initiate empty frame
        frame = np.zeros((200, 200, 3), dtype=np.uint8)

        on_screen = 0
        was_overlapped = False
        # For each of the objects currently on screen, add them at the new pos
        for track in tracks:
            is_on_screen, is_overlapping = update_frame(frame, track)

            if is_on_screen:
                on_screen += 1
            
            if is_overlapping:
                was_overlapped = True
        
        if was_overlapped:
            total_T += 1

        main_data["objects_per_frames"].append(on_screen)

        # # Add random noise to the frame
        # noise = np.random.randint(0, 25, (200, 200, 3), dtype=np.uint8)

        # frame = cv2.add(frame, noise)

        # Maybe add new tracks to the current tracks and remove finished ones
        for lane_index in range(len(generation_parameters["lanes"])):
            tracks, n_added, last_updated_lane = update_tracks(main_data["q_per_lane"][lane_index],
                                            generation_parameters["lanes"][lane_index],
                                            tracks,
                                            i,
                                            last_updated[lane_index],
                                            update_ratio=update_ratio[lane_index],
                                            max_objects=max_objects[lane_index])

            main_data["q"] += n_added
            main_data["q_per_lane"][lane_index] += n_added
            last_updated[lane_index] = last_updated_lane

        # Write and maybe display
        out.write(frame)

        if display:
            cv2.imshow('Frame', frame)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

    # Close all
    out.release()
    if display:
        cv2.destroyAllWindows()

    main_data["t"] = total_T / float(FPS * N_SECONDS)

    return main_data


def extract_motion_vectors(filename, display):
    """ Extract the motion vector and display if required.

    # Arguments:
        - filename: The file from which to extract the motion vectors.
        - display: If the extracted motion vectors should be displayed.
    """
    # load_gops(filename, number_of_gops, starting gop index)
    # mv_data = load_gops_size(filename, N_SECONDS, 0, *IMAGE_DIMS)
    mv_data, _ = extract_mvs.extract_mvs(filename)

    mv_data = np.transpose(mv_data, (1, 2, 0, 3))

    # Maybe display the motion vectors
    if display:
        for i in range(mv_data.shape[2]):
            frame = mv_data[:, :, i, :]

            frame = np.squeeze(frame)

            w, h, _ = frame.shape

            temp_frame = np.zeros((w, h, 3))
            temp_frame[:, :, :2] = frame

            # min/max frame convertion, but its only for display
            frame = convert(temp_frame)

            cv2.imshow('Motion Vectors', frame)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    return mv_data


def generate_residuals(output_dir: str, filename: str, motion_vectors: object,
                       display: bool):
    """ Function to generate the residual matrices.

    # Arguments:
        - filename: The RGB mp4 file to be used to generate the residuals.
        - motion_vectors: The associated motion vectors.
        - display: If the generated motion vectors should be displayed.
    """
    # First get the RGB frames required to compute the residuals
    rgb_frames = []

    cap = cv2.VideoCapture(filename)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            rgb_frames.append(frame)
        else:
            break
    cap.release()

    # Then compute the residuals and save them to a file
    residual_file = join(output_dir,
                         basename(filename).replace("rgb", "residual_qtv"))

    # We write the residuals to a mp4 video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(residual_file, fourcc, FPS,
                          (rgb_frames[0].shape[1], rgb_frames[0].shape[0]),
                          True)

    # Save the first residual frame
    # The residual data ranges from -255 to 255, for storage reason this is rescaled between [0 ; 255]
    # Because of that the zeros become 127, hence the times 127
    out.write(np.ones(rgb_frames[0].shape, dtype=np.uint8) * 127)

    # TODO Check the case were motion vectors are None
    splitted_motion_vectors = np.split(motion_vectors,
                                       int(FPS * N_SECONDS),
                                       axis=2)
    rows, cols, _, _ = splitted_motion_vectors[0].shape

    # Compute each of the residual matrices
    for index in range(1, len(splitted_motion_vectors)):

        # Current motion vector
        motion_vector = np.squeeze(splitted_motion_vectors[index])

        # If no motion vectors, set residuals to 0
        if np.min(motion_vector) == np.max(motion_vector) == 0:
            residual = np.ones(rgb_frames[0].shape, dtype=np.uint8) * 127
            out.write(residual)

            if display:
                cv2.imshow('Residuals', residual)

                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break
            continue
        else:
            residual = np.zeros(rgb_frames[0].shape, dtype=np.int16)

        # Get the frames that will be used for computation of the residuals
        reference_frame = rgb_frames[index - 1]
        current_frame = rgb_frames[index]

        # For each of the mvs, compute the residuals
        for r in range(rows):
            for c in range(cols):
                # Mapping motion vectors to correct type
                x, y = map(int, motion_vector[r, c])

                # In case reference block is partly out of the frame we will need to pad the reference block
                reference_r = max(r * 16 - y, 0)
                reference_r_max = min(r * 16 - y + 16, rows * 16)

                reference_c = max(c * 16 - x, 0)
                reference_c_max = min(c * 16 - x + 16, cols * 16)

                reference_block = reference_frame[
                    reference_r:reference_r_max,
                    reference_c:reference_c_max].astype(np.int16)

                # Only pad if out of frame as this operation is time consumming
                if not ((reference_c_max - reference_c) ==
                        (reference_r_max - reference_r) == 16):

                    pad = ((abs(min(r * 16 - y,
                                    0)), max(r * 16 - y + 16 - rows * 16, 0)),
                           (abs(min(c * 16 - x,
                                    0)), max(c * 16 - x + 16 - cols * 16,
                                             0)), (0, 0))
                    reference_block = np.pad(reference_block,
                                             pad,
                                             mode="constant",
                                             constant_values=0)

                # Update the residual reference image
                residual[r * 16:r * 16 + 16,
                         c * 16:c * 16 + 16] = reference_block[:16, :16]

        # Residual computation
        residual = current_frame - residual

        # Convertion to storable type (this imply a loss of information)
        residual = convert(residual, -255, 255, 0, 255, np.uint8)

        out.write(residual)

        if display:
            cv2.imshow('Residuals', residual)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

    out.release()
    if display:
        cv2.destroyAllWindows()


def save_remaining_data(output_dir, filename, main_data, motion_vectors):
    """ Save the remaining data into a hdf5 file.
        The save data contains the motion vectors, the final count of objects, the number of objects per frame, the label of the objects and their position.

    # Arguments:
        - output_dir: The directory where the generated data files will be saved.
        - filename: The name of the file to be saved.
        - main_data: All the data to be saved, motion vectors excluded.
        - motion_vectors: The motion vectors to save.
    """
    output_file = join(output_dir, "{:05d}_mv_qtv_data.hdf5".format(filename))

    with h5py.File(output_file, "w") as f:
        dset = f.create_dataset("motion_vectors",
                                data=motion_vectors,
                                compression="gzip")
        dset.attrs['q'] = main_data["q_per_lane"]
        dset.attrs['t'] = main_data["t"]
        dset.attrs['v'] = -1
        dset.attrs['day'] = "00"
        dset.attrs['hour'] = "00"
        dset.attrs['offset'] = 0
        dset.attrs['direction'] = "fuite"
        dset.attrs["missing_gop_frames"] = 0
        dset = f.create_dataset("objects_per_frames",
                                data=main_data["objects_per_frames"],
                                compression="gzip")


def generate_data(output_dir: str,
                  x_data: object,
                  y_data: object,
                  generation_parameters: Dict,
                  n_samples: int,
                  display: bool = False,
                  multiprocessing: bool = False):
    """ Final generation function, will create and save a dataset.
    For now all the available information is stored in the dataset.

        # Arguments:
            - output_dir: Were the dataset will be output.
            - x_data: The images to be used as objects for the Tracks.
            - y_data: The labels of the image to be used as objects for the Tracks.
            - generation_parameters: A dictionnary containing all the parameters required for the generation of the data (see Details).
            - n_samples: The number of samples to be generated.
            - display: If the generated objects should be displayed.
    """
    for lane_index in range(len(generation_parameters["lanes"])):
        generation_parameters["lanes"][lane_index]["images"] = x_data

    # Create legacy data path
    rgb_output_dir = join(output_dir, "rgb_data")
    compressed_output_dir = join(output_dir, "compressed_data")

    makedirs(rgb_output_dir)
    makedirs(compressed_output_dir)

    if multiprocessing:
        # No display in case of multiprocessing (not sure how it would work)
        display = False

        # with Pool(os.cpu_count()) as p:
        with Pool(os.cpu_count()) as p:
            # Generate the list of value to provide to the pool
            for iterator_count in tqdm(range(n_samples // 100)):
                data_inputs = [[
                    sample_index, rgb_output_dir, compressed_output_dir,
                    generation_parameters, display
                ] for sample_index in range(iterator_count *
                                            100, (iterator_count + 1) * 100)]

                p.starmap(generation_function, data_inputs)
    else:
        for sample_index in tqdm(range(n_samples)):
            generation_function(sample_index, rgb_output_dir,
                                compressed_output_dir, generation_parameters,
                                display)


def generation_function(sample_index, rgb_output_dir, compressed_output_dir,
                        generation_parameters, display):

    np.random.seed(None)

    filename = join(rgb_output_dir, "{:05d}_rgb_data.mp4".format(sample_index))
    temp_filename = join(rgb_output_dir,
                         "temp_{:05d}_rgb_data.mp4".format(sample_index))

    # First, generate and save the video file
    main_data = generate_video(filename, generation_parameters, display)

    # Convert file to correct format as opencv does not do that
    subprocess.call(
        'ffmpeg -hide_banner -loglevel panic -nostdin -y -i {} -g 25 -codec:v mpeg4 {}'
        .format(filename, temp_filename),
        shell=True)
    subprocess.call('mv {} {}'.format(temp_filename, filename), shell=True)

    # Extract the motion vectors
    try:
        motion_vectors = extract_motion_vectors(filename, display)
    except Exception as e:
        print(filename)
        raise e

    # # Generate and save the residual matrix
    # generate_residuals(compressed_output_dir, filename, motion_vectors,
    #                    display)

    # Save the remaining unsaved data into a h5py file.
    save_remaining_data(compressed_output_dir, sample_index, main_data,
                        motion_vectors)


def process_configuration_file(configuration_file):
    """ Reads the configuration file and returns a dictionnary containing all the parameters used for track generation.

    # Arguments:
        - configuration_file: The configuration file to be processed.

    # Returns:
        A dictionnary with all the parameters used for track generation.
    """
    with open(configuration_file) as json_file:
        data = json.load(json_file)

    return data


parser = argparse.ArgumentParser(
    description="Script to generate flux data based on the MNIST digit images."
)

parser.add_argument("output", help="Where to output the results.")
parser.add_argument(
    "configuration_file",
    help=
    "The json configuration file to be used for the generation of the videos.")
parser.add_argument("-d",
                    "--display",
                    help="Whether the generated videos should be displayed.",
                    action="store_true")
parser.add_argument("-dset",
                    "--dataset",
                    choices=['mnist', 'fashion_mnist'],
                    help="The dataset to use.",
                    default="mnist")
parser.add_argument("-ns",
                    "--number_of_samples",
                    help="The number of datapoints to generate.",
                    type=int,
                    default=5)
parser.add_argument("-mp",
                    "--multiprocessing",
                    help="If the generation should use multiprocessing",
                    action="store_true")

args = parser.parse_args()

# Extract the generation parameters
generation_parameters = process_configuration_file(args.configuration_file)

if args.dataset == "mnist":
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
elif args.dataset == "fashion_mnist":
    (x_train, y_train), (x_test,
                         y_test) = keras.datasets.fashion_mnist.load_data()

# Extract the training and validation sets
sep = int(len(x_train) * 0.8)

indexes = np.arange(len(x_train))

train_indexes = indexes[:sep]
validation_indexes = indexes[sep:]

x_val, y_val = x_train[validation_indexes], y_train[validation_indexes]
x_train, y_train = x_train[train_indexes], y_train[train_indexes]

# Compute the number of videos to generate for each of the sets
n_train_samples = args.number_of_samples
n_validation_samples = int(args.number_of_samples * 0.2)
n_test_samples = int(args.number_of_samples * 0.5)

# Create the train/validation/test output dirs
train_output_dir = join(args.output, "train")
validation_output_dir = join(args.output, "validation")
test_output_dir = join(args.output, "test")

makedirs(train_output_dir)
makedirs(validation_output_dir)
makedirs(test_output_dir)

# Generate the data
generate_data(train_output_dir, x_train, y_train, generation_parameters,
              n_train_samples, args.display, args.multiprocessing)
generate_data(validation_output_dir, x_val, y_val, generation_parameters,
              n_validation_samples, args.display, args.multiprocessing)
generate_data(test_output_dir, x_test, y_test, generation_parameters,
              n_test_samples, args.display, args.multiprocessing)
