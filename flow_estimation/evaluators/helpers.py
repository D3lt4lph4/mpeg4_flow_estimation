from random import randint

from os.path import join
from time import time

from typing import Dict

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator

import numpy as np

from sklearn.manifold import TSNE
from sklearn.metrics import r2_score

from scipy.interpolate import interp1d


def save_tsne(xs_t, xt_t, xs_t_label, xt_t_label, title, output_dir, n_labels=None):

    xs = xs_t  # [:2500]
    xt = xt_t  # [:2500]
    xs_label = xs_t_label  # [:2500]
    xt_label = xt_t_label  # [:2500]

    # Combine the extracted representations
    combined_embedded = np.vstack([xs, xt])
    combined_labels = np.vstack([xs_label, xt_label])

    # Get the index of the last source point
    s_index = len(xs_label)
    t_index = len(xt_label)

    if n_labels is None:

        unique_labels = np.unique(list(map(int, combined_labels)))

        colors = cm.brg(np.linspace(0, 1, max(unique_labels)+1))
    else:
        colors = cm.brg(np.linspace(0, 1, n_labels))

    # Create the t-sne module
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)

    # Fit the data-points to the t-sne representation
    source_only_tsne = tsne.fit_transform(combined_embedded)

    # Create the figure
    plt.figure(figsize=(15, 15))

    # # Get the index of the last source point
    # s_index = len(xs_label)

    source_colors = [colors[int(xs_label[i, 0])] for i in range(s_index)]
    target_colors = [colors[int(xt_label[i, 0])] for i in range(t_index)]

    # Print the points on the figure
    plt.scatter(source_only_tsne[:s_index, 0],
                source_only_tsne[:s_index, 1],
                c=source_colors,
                s=50,
                edgecolors='b',
                alpha=0.5,
                marker='o',
                cmap=cm.jet,
                label='source')
    plt.scatter(source_only_tsne[s_index:, 0],
                source_only_tsne[s_index:, 1],
                c=target_colors,
                s=50,
                edgecolors='b',
                alpha=0.5,
                marker='X',
                cmap=cm.jet,
                label='target')

    plt.axis('off')
    plt.legend(loc='best')
    plt.title(title)

    plt.savefig(join(output_dir, title + '.png'),
                bbox_inches='tight',
                pad_inches=0,
                format='png')

    plt.close()

    # save_hexbin(source_only_tsne[:s_index,0], source_only_tsne[:s_index,1], output_dir, title, "source")

    # save_hexbin(source_only_tsne[s_index:,0], source_only_tsne[s_index:,1], output_dir, title, "target")


def extract_sample_from_matrix(X,
                               max_iter=1000,
                               sample_size=200,
                               keep_all=-1,
                               points_to_ignore=None):
    """ Randomly samples separate points from a given matrix.

    # Arguments:
        - X: The input matrix to sample the point from.
        - max_iter: The max number of time the
        - sample_size:
        - keep_all: The axes for which all the points should be kept
    """
    samples = []
    tested_point = []
    points_to_ignore = [] if points_to_ignore is None else points_to_ignore

    matrix_shape = X.shape

    dim_range = list(range(len(matrix_shape)))

    keep_all = [keep_all] if isinstance(keep_all, int) else keep_all

    # Get the real axes values for the keep all
    keep_all = [dim_range[pos] for pos in keep_all]

    while len(samples) < sample_size and len(tested_point) < max_iter:

        point_position = []
        # Generate the random point
        for value in dim_range:
            # Add the axis where we keep all the values
            if value in keep_all:
                point_position.append(slice(None, None, None))
                continue
            point_position.append(randint(0, matrix_shape[value] - 1))

        # If already tested point, continue
        if point_position in tested_point:
            continue
        else:
            tested_point.append(point_position)

        # Check if point to ignore
        if X[point_position] in points_to_ignore:
            continue
        else:
            samples.append(X[point_position][:2])

    return np.array(samples)


def extract_sample_from_matrix_v2(X, sample_size=200):
    """ Randomly samples separate points from a given matrix.

    # Arguments:
        - X: The input matrix to sample the point from.
        - max_iter: The max number of time the
        - sample_size:
        - keep_all: The axes for which all the points should be kept
    """
    samples = []
    matrix_shape = X.shape

    for i in range(sample_size):
        w = randint(0, matrix_shape[0] - 1)
        h = randint(0, matrix_shape[1] - 1)
        ts = randint(0, matrix_shape[2] - 1)
        ts_2 = randint(0, matrix_shape[3] - 1)

        samples.append(X[w, h, ts, ts_2, :])

    return np.array(samples)


def save_histogram(X,
                   key_indexes,
                   output_directory,
                   value_name,
                   data_range=range(0, 30, 1)):
    """ Save the histograms of the values given. Multiple indexes can be provided to display multiple histograms.

    # Arguments:
        - X: The data to plot.
        - key_indexes: A dictionnary contaning arrays with the indexes of the points to plot for the given key. For instance, keys could represent different time segment to plot for a list of data points.
        - output_directory: The directory where the graphs should be saved.
        - value_name: The name of the value being plotted.
        - data_range: The range used for the generation of the bins to represent the data.

    """
    # Generate the figure
    plt.figure()

    # For each of the index key, plot the histogram of the associated datapoint.
    for key in key_indexes:
        indexes = key_indexes[key]

        n, x = np.histogram(X[indexes], bins=data_range)

        bin_centers = 0.5 * (x[1:] + x[:-1])

        f = interp1d(bin_centers, n, kind='cubic')

        xnew = np.linspace(bin_centers[0],
                           bin_centers[-1],
                           num=100,
                           endpoint=True)

        plt.plot(xnew, f(xnew), label="{}".format(key))

    plt.legend(loc="upper right")
    plt.xlabel("{}".format(value_name))
    plt.ylabel("Cumsum")

    plt.savefig(join(output_directory, "{}".format(value_name)),
                dpi=300,
                bbox_inches='tight')
    plt.close()


def save_2d_distribution(X: object, key_indexes: Dict, output_directory: str,
                         id: str,
                         data_type: str = "cartesian"):
    """ Plot the 2d distribution of the data.

    # Arguments:
        - X: The data to be plotted.
        - key_indexes: A dictionnary containing the indexes to plot for each of the keys
        - output_directory: Where the plot should be saved.
        - id: The id that should be used for the name of the file to save.
        - data_type: The type of the data, one of ['polar', 'polar_module', 'cartesian'].
    """
    if data_type == 'cartesian':
        save_cartesian_2d_distribution(X, key_indexes, output_directory, id)
    elif data_type == 'polar':
        save_polar_2d_distribution(X, key_indexes, output_directory, id)
    elif data_type == 'polar_module':
        raise RuntimeError(
            "Plot saving not implemented for polar module coordinates.")
    else:
        raise RuntimeError(
            "The data type {} is not supported.".format(data_type))


def save_cartesian_2d_distribution(X: object, key_indexes: Dict, output_directory: str, id: str):
    """ Plot the 2d distribution of the data for cartesian representation .

    # Arguments:
        - X: The data to be plotted.
        - key_indexes: A dictionnary containing the indexes to plot for each of the keys
        - output_directory: Where the plot should be saved.
        - id: The id that should be used for the name of the file to save.
    """
    fig = plt.figure()

    for key in key_indexes:
        ax = fig.add_subplot(111)
        mv_key = X[key_indexes[key]].reshape((-1, X.shape[-1]))

        x = mv_key[:, 0]
        y = mv_key[:, 1]

        hist, xedges, yedges = np.histogram2d(x,
                                              y,
                                              bins=40,
                                              range=[[-20, 20], [-20, 20]])

        # ax.plot_surface(xpos, ypos, hist)
        img = ax.imshow(np.log(hist), cmap="viridis")
        fig.colorbar(img)

        ax.set_title("Motion vectors distribution: {}".format(key))
        plt.savefig(join(output_directory,
                         "{}_{}_mv_distribution.png".format(key, id)),
                    dpi=300,
                    bbox_inches='tight')
        plt.clf()
    plt.close()


def save_polar_2d_distribution(X: object, key_indexes: Dict, output_directory: str, id: str):
    """ Plot the 2d distribution of the data for polar representation .

    # Arguments:
        - X: The data to be plotted.
        - key_indexes: A dictionnary containing the indexes to plot for each of the keys
        - output_directory: Where the plot should be saved.
        - id: The id that should be used for the name of the file to save.
    """
    fig = plt.figure()

    for key in key_indexes:
        ax = fig.add_subplot(111, projection="polar")
        mv_key = X[key_indexes[key]].reshape((-1, X.shape[-1]))

        x = mv_key[:, 0]
        y = mv_key[:, 1]

        # define binning
        rbins = np.linspace(0, max(20, np.max(y).astype(np.int)), 30)
        abins = np.linspace(0, 2*np.pi, 60)

        # calculate histogram
        hist, _, _ = np.histogram2d(y, x, bins=(abins, rbins))
        A, R = np.meshgrid(abins, rbins)

        # hist[hist == 0] = 1

        img = ax.pcolormesh(A, R, np.log(hist.T), cmap="plasma")
        fig.colorbar(img)

        ax.set_title("Motion vectors distribution: {}".format(key))
        plt.savefig(join(output_directory,
                         "{}_{}_mv_distribution.png".format(key, id)),
                    dpi=300,
                    bbox_inches='tight')
        plt.clf()
    plt.close()


def save_hexbin(y_true, y_pred, output_directory, id, value_name):
    """ Plot and save the hexbin heatmap of the real vs predicted values.

    # Arguments:
        - y_true: The true values of the data.
        - y_pred: The predicted values of the data.
        - output_directory: Where to output the results.
        - id: The id of the plot being created, used for name saving.
        - value_name: The name of the value being plotted (Q, T, ...).
    """
    # Create the figure and prepare the labels
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.title("{} vs Real {}".format(value_name, value_name))
    plt.xlabel("Real {}".format(value_name))
    plt.ylabel("{}".format(value_name))

    # Create the hexbin heatmap
    hb = plt.hexbin(y_true, y_pred, gridsize=30, bins='log', cmap='inferno')

    # Add the scale
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('counts (log)')

    # Save the figure
    plt.savefig(
        join(output_directory, "{}_{}_hexbin.png".format(id, value_name)))
    plt.close()


def save_plot(y_true, y_pred, output_directory, value_name, id):
    """ Save the plot of the Real vs Predicted datapoints.

    # Arguments:
        - y_true: The true values of the data.
        - y_pred: The predicted values of the data.

    """
    fig = plt.figure()
    ax = fig.gca()

    plt.xlim([-1, 18])
    plt.ylim([-1, 18])

    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)

    plt.title("Predicted {} vs Real {}".format(
        value_name, value_name), fontsize=22)
    plt.xlabel("Real {}".format(value_name), fontsize=17)
    plt.ylabel("Predicted {}".format(value_name), fontsize=17)

    # Plot the predictions
    plt.scatter(y_true, y_pred)

    # Get the ranges to draw the x = y line
    y_range = plt.ylim()
    x_range = plt.xlim()

    draw_range = []

    if x_range[0] < y_range[0]:
        draw_range.append(y_range[0] - 2)
    else:
        draw_range.append(x_range[0] - 2)

    if x_range[1] < y_range[1]:
        draw_range.append(x_range[1] + 2)
    else:
        draw_range.append(y_range[1] + 2)

    plt.plot(draw_range, draw_range, color="red")

    # Draw the x and y axis
    plt.hlines(0, *x_range, linestyles="dotted")
    plt.vlines(0, *y_range, linestyles="dotted")

    plt.xlim(x_range)
    plt.ylim(y_range)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Save the figure
    plt.savefig(
        join(output_directory, "{}_{}_scatter.png".format(id, value_name)), bbox_inches="tight")
    plt.close()


def save_time_plot(y_true, y_pred, segment_index, output_directory, value_name,
                   id):
    """ Generate a plot of real vs predicted values over time.
    Multiple time instances can be plotted, dashed lines will be used to separate them.
    The mae, r, r2 values will also be computed and plotted.

    # Arguments:
        - y_true: The real values, will be plotted in black.
        - y_pred: The predicted values, will be plotted red.
        - segment_index: A dictionary containing the index in time order for each of the time segment.
        - output_dir: The output directory for the plot.
    """
    # Prepare the figure
    fig = plt.figure(figsize=(9,3))

    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)

    number_of_point = 250
    pred_size = len(y_true)
    step = pred_size // number_of_point

    # fig.set_size_inches(15, 5)

    # Computing the MAE
    mae = np.mean(np.abs(y_true - y_pred))

    # Computing the correlation coefficient
    r = np.corrcoef(y_true.T, y_pred.T)

    # Computing the coefficient of determination
    r2_dict = {}
    global_r2 = 0
    for key in segment_index:
        r2_dict[key] = r2_score(y_true[segment_index[key]],
                                y_pred[segment_index[key]])
        global_r2 += r2_dict[key]

    # Sort the key to have the data in the correct order
    time_sorted_segments_key = sorted(segment_index.keys())
    global_r2 = global_r2 / len(time_sorted_segments_key)

    # Create the separators for the time segments and sort the data
    separators = [0]
    separators_2 = [0]
    y_pred_sorted = np.zeros_like(y_pred)
    y_true_sorted = np.zeros_like(y_true)

    for time_key in time_sorted_segments_key:
        indexes = segment_index[time_key]
        separators.append(len(indexes) + separators[-1])
        separators_2.append(len(indexes) // step + separators_2[-1])
        y_pred_sorted[separators[-2]:separators[-1], :] = y_pred[indexes]
        y_true_sorted[separators[-2]:separators[-1], :] = y_true[indexes]

    # max_range = int(np.max(np.maximum(y_pred_sorted, y_true_sorted)) + 20)
    max_range = 15

    # Add the vertical separators
    for i, sep in enumerate(separators_2[1:-1]):
        # Put the text in the middle of the two separators
        text_position = separators_2[i] + (sep - separators_2[i]) / 2

        hour = int(time_sorted_segments_key[i].split(":")[1])

        if hour == 8:
            time_value = "8-10 am"
        elif hour == 12:
            time_value = "noon - 1 pm"
        elif hour == 18:
            time_value = "6-8 pm"
        elif hour == 22:
            time_value = "10-11 pm"

        # time_value = ":".join(time_sorted_segments_key[i].split(":")[1:])

        plt.vlines(x=sep, ymin=-10, ymax=100, linestyles="dashed")
        plt.text(text_position,
                 max_range - 1.5,
                 "{}".format(time_value),
                 ha='center',
                 fontsize=12)
    else:
        text_position = separators_2[-2] + \
            (separators_2[-1] - separators_2[-2]) / 2
        # time_value = ":".join(time_sorted_segments_key[-1].split(":")[1:])
        hour = int(time_sorted_segments_key[-1].split(":")[1])

        if hour == 8:
            time_value = "8-10 am"
        elif hour == 12:
            time_value = "noon - 1 pm"
        elif hour == 18:
            time_value = "6-8 pm"
        elif hour == 22:
            time_value = "10-11 pm"
        plt.text(text_position,
                 max_range - 1.5,
                 "{}".format(time_value),
                 ha='center',
                 fontsize=12)

    # add the data
    plt.plot(y_pred_sorted[::step], label="Predicted")
    plt.plot(y_true_sorted[::step], label="Real")
    plt.legend(loc="upper left", fontsize=12)

    plt.xlabel('Time', fontsize=18)
    plt.ylabel('{} value'.format(value_name), fontsize=18)

    plt.xlim([0, len(y_pred_sorted[::step])])
    plt.ylim((0, max_range))

    # fig.suptitle(
    #     "{} Real/Predicted over time, MAE = {:.3f}, R = {:.3f}, R2 = {:.3f}".
    #     format(value_name, mae, r[0, 1], global_r2))

    # Save the figure
    plt.savefig(
        join(output_directory, "{}_{}_over_time.png".format(id, value_name)), bbox_inches='tight')

    # Closing current plot to avoid errors later
    plt.close()


def save_error_plot(y_true, y_pred, output_directory, value_name, id):
    """ Helper function that generates an error plot between two values.

    # Arguments:
        - y_true: The real values, used as X axis.
        - y_pred: The predicted values, used to compute the error.
        - output_dir: Where to output the results.
        - value_name: The name of the value we are working on (Q, T, ...).
        - id: An id used in the name of the save file.
    """
    # First compute the error between true and predicted values
    data_error = np.abs(y_true - y_pred)

    fig = plt.figure()
    fig.set_size_inches(6.4, 4.8)
    ax = fig.add_subplot(111)

    hb = plt.hexbin(y_true,
                    data_error,
                    gridsize=30,
                    bins='log',
                    cmap='inferno')

    # Add the scale
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('counts (log)')

    plt.xlabel('{} value'.format(value_name), fontsize=18)
    plt.ylabel('Error', fontsize=16)
    plt.title("Error Hexbin for the predicted {} values".format(value_name))
    plt.savefig(
        join(output_directory, "{}_{}_error_hexbin.png".format(id,
                                                               value_name)))

    plt.close()
