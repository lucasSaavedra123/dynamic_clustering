import pandas as pd
import matplotlib.pyplot as plt

from CONSTANTS import *

from argparse import ArgumentParser

def generate_colors_for_cluster_ids(max_cluster_id):
    color_list = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'olive', 'cyan', 'black', 'magenta', 'navy', 'lime', 'darkred']
    
    id_to_color = {}
    id_to_color = {0: 'grey'}

    for cluster_id in range(1, max_cluster_id+1):
        id_to_color[cluster_id] = color_list[cluster_id % len(color_list)]

    return id_to_color

parser = ArgumentParser()
parser.add_argument("-f", "--file", dest="filename")
parser.add_argument("-d", "--dimension", default='3d', dest="dimension")
parser.add_argument("-c", "--with-clustering", default=False, dest="with_clustering")
parser.add_argument("-s", "--save_plots", default=False, dest="save_plots")
parser.add_argument("-r", "--filter", default=False, dest="filter")

args = parser.parse_args()

dataset = pd.read_csv(args.filename)

print(f"Average: {len(dataset)/max(dataset[FRAME_COLUMN_NAME])}")

projection = args.dimension
with_clustering = args.with_clustering
filter_flag = args.filter

if with_clustering:
    dataset[CLUSTER_ID_COLUMN_NAME] = dataset[CLUSTER_ID_COLUMN_NAME].map(generate_colors_for_cluster_ids(max(dataset[CLUSTER_ID_COLUMN_NAME])))
if filter_flag:
    dataset = dataset[dataset[CLUSTERIZED_COLUMN_NAME] == 1]


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
        plt.savefig(f"{args.filename}.jpg")
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
        plt.savefig(f"{args.filename}_3d.jpg")
    else:
        plt.show()