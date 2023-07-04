import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, adjusted_rand_score
import seaborn as sns
import numpy as np

from CONSTANTS import *

from argparse import ArgumentParser
from argparse import BooleanOptionalAction

def validate_range(name, a_range):
    if a_range != []:
        if len(a_range) != 2:
            raise ValueError(f"{name} should contain two values")
        elif a_range[0] > a_range[1]:
            raise ValueError(f"{name}[0] < {name}[1]")

def validate_arguments(args):
    dict_of_ranges = {
        '--frame_range': args.frame_range,
        '--roi_x': args.roi_x,
        '--roi_y': args.roi_y
    }

    for dict_key in dict_of_ranges:
        validate_range(dict_key, dict_of_ranges[dict_key])

def filter_dataset_from_arguments(args, dataset):
    dict_of_ranges = {
        FRAME_COLUMN_NAME: args.frame_range,
        X_POSITION_COLUMN_NAME: args.roi_x,
        Y_POSITION_COLUMN_NAME: args.roi_y
    }

    for dict_key in dict_of_ranges:
        if dict_of_ranges[dict_key] != []:
            dataset = dataset[dict_of_ranges[dict_key][0] < dataset[dict_key]]
            dataset = dataset[dataset[dict_key] < dict_of_ranges[dict_key][1]]

    return dataset

def generate_colors_for_cluster_ids(cluster_ids):
    color_list = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'olive', 'cyan', 'black', 'magenta', 'navy', 'lime', 'darkred']
    
    id_to_color = {}
    id_to_color = {0: 'grey'}

    if 0 in cluster_ids:
        cluster_ids.remove(0)

    if len(cluster_ids) == 1:
        print("Number Of Clusters:", len(cluster_ids))

    for index, cluster_id in enumerate(cluster_ids):
        id_to_color[cluster_id] = color_list[index % len(color_list)]

    return id_to_color

parser = ArgumentParser()
parser.add_argument("-f", "--file", dest="filename")
parser.add_argument("-p", "--projection", default='3d', dest="projection")
parser.add_argument("-c", "--with-clustering", default=False, dest="with_clustering", action=BooleanOptionalAction)
parser.add_argument("-s", "--save_plots", default=False, dest="save_plots", action=BooleanOptionalAction)
parser.add_argument("-r", "--filter_flag", default=False, dest="filter_flag", action=BooleanOptionalAction)
parser.add_argument("-b", "--binary_clustering", default=False, dest="binary_clustering", action=BooleanOptionalAction)
parser.add_argument("-i", "--predicted", default=False, dest="predicted", action=BooleanOptionalAction)
parser.add_argument("-a", "--show_performance", default=False, dest="show_performance", action=BooleanOptionalAction)
parser.add_argument("-m", "--from_magic", default=False, dest="from_magic", action=BooleanOptionalAction)

#Range Arguments
parser.add_argument('-fr', '--frame_range', type=int, nargs='+', default=[])
parser.add_argument('-rx', '--roi_x', type=float, nargs='+', default=[])
parser.add_argument('-ry', '--roi_y', type=float, nargs='+', default=[])

args = parser.parse_args()

validate_arguments(args)

dataset = pd.read_csv(args.filename)

if args.from_magic:
    dataset = dataset.rename(columns={
        MAGIK_X_POSITION_COLUMN_NAME: X_POSITION_COLUMN_NAME,
        MAGIK_Y_POSITION_COLUMN_NAME: Y_POSITION_COLUMN_NAME,
    })

    if args.with_clustering:
        if not args.binary_clustering:
            dataset = dataset.rename(columns={
                MAGIK_LABEL_COLUMN_NAME: CLUSTER_ID_COLUMN_NAME,
                MAGIK_LABEL_COLUMN_NAME_PREDICTED: CLUSTER_ID_COLUMN_NAME+"_predicted",
            })
        else:
            dataset = dataset.rename(columns={
                MAGIK_LABEL_COLUMN_NAME: CLUSTERIZED_COLUMN_NAME,
                MAGIK_LABEL_COLUMN_NAME_PREDICTED: CLUSTERIZED_COLUMN_NAME+"_predicted",
            })

dataset = filter_dataset_from_arguments(args, dataset)

print(f"Average: {len(dataset)/(max(dataset[FRAME_COLUMN_NAME] + 1))}")

if len(set(dataset[FRAME_COLUMN_NAME].values.tolist())) == 1:
    args.projection = '2d'

if args.predicted and args.show_performance and args.with_clustering:
    if args.binary_clustering:
        print("F1 Score:", f1_score(dataset[CLUSTERIZED_COLUMN_NAME], dataset[CLUSTERIZED_COLUMN_NAME + '_predicted']))
    else:
        print("ARI:", adjusted_rand_score(dataset[CLUSTER_ID_COLUMN_NAME], dataset[CLUSTER_ID_COLUMN_NAME + '_predicted']))

if args.with_clustering:
    if args.predicted and args.binary_clustering:
        column_to_pick = CLUSTERIZED_COLUMN_NAME + '_predicted'
    elif args.predicted and (not args.binary_clustering):
        column_to_pick = CLUSTER_ID_COLUMN_NAME + '_predicted'
    elif (not args.predicted) and args.binary_clustering:
        column_to_pick = CLUSTERIZED_COLUMN_NAME
    elif (not args.predicted) and (not args.binary_clustering):
        column_to_pick = CLUSTER_ID_COLUMN_NAME
    else:
        raise Exception("Invalid combination of parameters")

    cluster_ids = list(set(dataset[column_to_pick].values.tolist()))

    if args.binary_clustering:
        dataset['COLOR_COLUMN'] = dataset[column_to_pick].map(generate_colors_for_cluster_ids(cluster_ids))
    else:
        dataset['COLOR_COLUMN'] = dataset[column_to_pick].map(generate_colors_for_cluster_ids(cluster_ids))

if args.filter_flag:
    original_number_of_localizations = len(dataset)
    if args.predicted:
        dataset = dataset[dataset[CLUSTERIZED_COLUMN_NAME + '_predicted'] == 1]
    else:
        dataset = dataset[dataset[CLUSTERIZED_COLUMN_NAME] == 1]
    new_number_of_localizations = len(dataset)

    print(f"Number of Localizations Removed: {(original_number_of_localizations-new_number_of_localizations)/original_number_of_localizations}")

if args.projection == '3d' or args.save_plots:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if args.with_clustering:
        ax.scatter(dataset[X_POSITION_COLUMN_NAME], dataset[TIME_COLUMN_NAME], dataset[Y_POSITION_COLUMN_NAME], c=dataset['COLOR_COLUMN'], s=1)
    else:
        ax.scatter(dataset[X_POSITION_COLUMN_NAME], dataset[TIME_COLUMN_NAME], dataset[Y_POSITION_COLUMN_NAME], s=1)

    ax.set_xlabel('x[um]')
    ax.set_ylabel('t[s]')
    ax.set_zlabel('y[um]')

    if args.save_plots:
        plt.savefig(f"{args.filename}_3d.jpg", dpi=300)
    else:
        plt.show()

if args.projection == '2d' or args.save_plots:
    fig = plt.figure()
    ax = fig.add_subplot()

    if args.with_clustering:
        ax.scatter(dataset[X_POSITION_COLUMN_NAME], dataset[Y_POSITION_COLUMN_NAME], c=dataset['COLOR_COLUMN'], s=1)
    else:
        ax.scatter(dataset[X_POSITION_COLUMN_NAME], dataset[Y_POSITION_COLUMN_NAME], s=1)

    ax.set_xlabel('x[um]')
    ax.set_ylabel('y[um]')

    if args.roi_x != []:
        ax.set_xlim(args.roi_x[0], args.roi_x[1])
    if args.roi_y != []:
        ax.set_ylim(args.roi_y[0], args.roi_y[1])

    ax.set_aspect('equal')

    if args.save_plots:
        plt.savefig(f"{args.filename}_2d.jpg", dpi=300)
    else:
        plt.show()

if args.show_performance:
    confusion_mat = confusion_matrix(y_true=dataset[CLUSTERIZED_COLUMN_NAME].values.tolist(), y_pred=dataset[CLUSTERIZED_COLUMN_NAME+"_predicted"].values.tolist())
    confusion_mat = np.round(confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis], 2)

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
        plt.savefig(f"{args.filename}_confusion_matrix.jpg", dpi=300)
    else:
        plt.show()
