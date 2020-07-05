#!/usr/bin/env python3

"""
<< predict_mean.py >>
"""

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import argparse
from sys import argv, exit

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on the bottom.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

if __name__ == '__main__':
    """
    """
    # Print docstring if only the name of the script is given
    if len(argv) < 2:
        print(__doc__)
        exit(0)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Options for predict_mean.py", add_help=True)

    required = parser.add_argument_group("required arguments")
    required.add_argument('-cu','--coal_units', action="store", type=float, default=1.0,
                            metavar='\b', help="branch scaling in coalescent units")

    args = parser.parse_args()
    cu   = args.coal_units

    # Read in testing data and load model
    data = np.load('../processed_data/hyde_cnn_mean_data_{}.npz'.format(cu))
    xtest,ytest = data['xtest'],data['ytest']
    del data

    model = load_model('hyde_cnn_mean_{}.mdl'.format(cu))
    pred = model.predict(xtest)
    pred_cat = [i.argmax() for i in pred]

    # Get prediction accuracy
    print(classification_report([np.argmax(ytest[i,:]) for i in range(ytest.shape[0])],
                      pred_cat, digits=3, target_names=['no_hyb', 'hyb_sp', 'admix', 'admix_mig']))

    # Make and plot confusion matrix
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots()
    cm = np.array(confusion_matrix([np.argmax(ytest[i,:]) for i in range(ytest.shape[0])],pred_cat))
    cm_norm = [cm[i,:]/np.sum(cm[i,:]) for i in range(cm.shape[0])]
    im, cbar = heatmap(np.array(cm_norm), ['no_hyb', 'hyb_sp', 'admix', 'admix_mig'],
                       ['no_hyb', 'hyb_sp', 'admix', 'admix_mig'], ax=ax,
                       cmap="Greys", vmin=0, vmax=1, cbarlabel="Accuracy")
    texts = annotate_heatmap(im, valfmt="{x:.3f}")
    fig.tight_layout()
    plt.savefig("hyde_cnn_mean_confusion-matrix_{}.svg".format(cu))
    plt.savefig("hyde_cnn_mean_confusion-matrix_{}.png".format(cu))
    plt.show()
