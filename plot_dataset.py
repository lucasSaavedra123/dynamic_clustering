import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

from CONSTANTS import *

from argparse import ArgumentParser
from argparse import BooleanOptionalAction


def generate_colors_for_cluster_ids(max_cluster_id):
    color_list = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'olive', 'cyan', 'black', 'magenta', 'navy', 'lime', 'darkred']
    
    id_to_color = {}
    id_to_color = {0: 'grey'}

    for cluster_id in range(1, int(max_cluster_id)+1):
        id_to_color[cluster_id] = color_list[cluster_id % len(color_list)]

    return id_to_color

parser = ArgumentParser()
parser.add_argument("-f", "--file", dest="filename")
parser.add_argument("-d", "--dimension", default='3d', dest="dimension")
parser.add_argument("-c", "--with-clustering", default=False, dest="with_clustering", action=BooleanOptionalAction)
parser.add_argument("-s", "--save_plots", default=False, dest="save_plots", action=BooleanOptionalAction)
parser.add_argument("-r", "--filter", default=False, dest="filter", action=BooleanOptionalAction)
parser.add_argument("-b", "--binary_clustering", default=False, dest="binary_clustering", action=BooleanOptionalAction)
parser.add_argument("-p", "--predicted", default=False, dest="predicted", action=BooleanOptionalAction)
parser.add_argument("-a", "--show_confusion_matrix", default=None, dest="show_confusion_matrix", action=BooleanOptionalAction)
parser.add_argument("-mn", "--min_frame", default=None, dest="min_frame")
parser.add_argument("-mx", "--max_frame", default=None, dest="max_frame")

args = parser.parse_args()

dataset = pd.read_csv(args.filename)

if args.min_frame is not None and args.max_frame is not None:
    dataset = dataset[int(args.min_frame) < dataset[FRAME_COLUMN_NAME]]
    dataset = dataset[dataset[FRAME_COLUMN_NAME] < int(args.max_frame)]


if MAGIK_LABEL_COLUMN_NAME_PREDICTED in dataset.columns:
    if not len(sorted(np.unique(dataset[MAGIK_LABEL_COLUMN_NAME].values).tolist())) == [0,1]:
        dataset = dataset.rename(columns={
            MAGIK_X_POSITION_COLUMN_NAME: X_POSITION_COLUMN_NAME,
            MAGIK_Y_POSITION_COLUMN_NAME: Y_POSITION_COLUMN_NAME,
            MAGIK_LABEL_COLUMN_NAME: CLUSTER_ID_COLUMN_NAME,
            MAGIK_LABEL_COLUMN_NAME_PREDICTED: CLUSTER_ID_COLUMN_NAME+"_predicted",
        })
    else:
        dataset = dataset.rename(columns={
            MAGIK_X_POSITION_COLUMN_NAME: X_POSITION_COLUMN_NAME,
            MAGIK_Y_POSITION_COLUMN_NAME: Y_POSITION_COLUMN_NAME,
            MAGIK_LABEL_COLUMN_NAME: CLUSTERIZED_COLUMN_NAME,
            MAGIK_LABEL_COLUMN_NAME_PREDICTED: CLUSTERIZED_COLUMN_NAME+"_predicted",
        })

print(f"Average: {len(dataset)/max(dataset[FRAME_COLUMN_NAME])}")

projection = args.dimension
with_clustering = args.with_clustering
filter_flag = args.filter
binary_clustering = args.binary_clustering
predicted = args.predicted
show_confusion_matrix = args.show_confusion_matrix

if with_clustering:
    if predicted and binary_clustering:
        column_to_pick = CLUSTERIZED_COLUMN_NAME + '_predicted'
    elif predicted and (not binary_clustering):
        column_to_pick = CLUSTER_ID_COLUMN_NAME + '_predicted'
    elif (not predicted) and binary_clustering:
        column_to_pick = CLUSTERIZED_COLUMN_NAME
    elif (not predicted) and (not binary_clustering):
        column_to_pick = CLUSTER_ID_COLUMN_NAME
    else:
        raise Exception("Invalid combination of parameters")

    if binary_clustering:
        dataset[CLUSTER_ID_COLUMN_NAME] = dataset[column_to_pick].map(generate_colors_for_cluster_ids(max(dataset[column_to_pick])))
    else:
        dataset[CLUSTER_ID_COLUMN_NAME] = dataset[column_to_pick].map(generate_colors_for_cluster_ids(max(dataset[column_to_pick])))

if filter_flag:
    original_number_of_localizations = len(dataset)
    if predicted:
        dataset = dataset[dataset[CLUSTERIZED_COLUMN_NAME+ '_predicted'] == 1]
    else:
        dataset = dataset[dataset[CLUSTERIZED_COLUMN_NAME] == 1]
    new_number_of_localizations = len(dataset)

    print(f"Number of Localizations Removed: {(original_number_of_localizations-new_number_of_localizations)/original_number_of_localizations}")

if projection == '3d' or args.save_plots:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if with_clustering:
        ax.scatter(dataset[X_POSITION_COLUMN_NAME], dataset[TIME_COLUMN_NAME], dataset[Y_POSITION_COLUMN_NAME], c=dataset[CLUSTER_ID_COLUMN_NAME], s=1)
    else:
        ax.scatter(dataset[X_POSITION_COLUMN_NAME], dataset[TIME_COLUMN_NAME], dataset[Y_POSITION_COLUMN_NAME], s=1)

    ax.set_xlabel('x[um]')
    ax.set_ylabel('t[s]')
    ax.set_zlabel('y[um]')

    if args.save_plots:
        plt.savefig(f"{args.filename}_3d.jpg")
    else:
        plt.show()

if projection == '2d' or args.save_plots:
    fig = plt.figure()
    ax = fig.add_subplot()

    if with_clustering:
        ax.scatter(dataset[X_POSITION_COLUMN_NAME], dataset[Y_POSITION_COLUMN_NAME], c=dataset[CLUSTER_ID_COLUMN_NAME], s=1)
    else:
        ax.scatter(dataset[X_POSITION_COLUMN_NAME], dataset[Y_POSITION_COLUMN_NAME], s=1)

    ax.set_xlabel('x[um]')
    ax.set_ylabel('y[um]')

    if args.save_plots:
        plt.savefig(f"{args.filename}_2d.jpg")
    else:
        plt.show()

if show_confusion_matrix:
    confusion_mat = confusion_matrix(y_true=dataset[CLUSTERIZED_COLUMN_NAME].values.tolist(), y_pred=dataset[CLUSTERIZED_COLUMN_NAME+"_predicted"].values.tolist())
    confusion_mat = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]

    if confusion_mat.shape == (1,1):
        confusion_mat = np.array([[1, 0], [0, 0]])

    labels = ["Non-Clusterized", "Clusterized"]

    confusion_matrix_dataframe = pd.DataFrame(data=confusion_mat, index=labels, columns=labels)
    sns.set(font_scale=1.5)
    color_map = sns.color_palette(palette="Blues", n_colors=7)
    sns.heatmap(data=confusion_matrix_dataframe, annot=True, annot_kws={"size": 15}, cmap=color_map)

    plt.title(f'Confusion Matrix')
    plt.rcParams.update({'font.size': 15})
    plt.ylabel("Ground truth", fontsize=15)
    plt.xlabel("Predicted label", fontsize=15)

    if args.save_plots:
        plt.savefig(f"{args.filename}_confusion_matrix.jpg")
    else:
        plt.show()
