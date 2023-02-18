import pandas as pd
import matplotlib.pyplot as plt

from CONSTANTS import *

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-f", "--file", dest="filename")
parser.add_argument("-d", "--dimension", default='3d', dest="dimension")
parser.add_argument("-c", "--with-clustering", default=False, dest="with_clustering")

args = parser.parse_args()

dataset = pd.read_csv(args.filename)

print(f"Average: {len(dataset)/max(dataset[FRAME_COLUMN_NAME])}")

projection = args.dimension
with_clustering = args.with_clustering

if with_clustering:
    dataset[CLUSTERIZED_COLUMN_NAME] = dataset[CLUSTERIZED_COLUMN_NAME].map({0: 'grey', 1: 'red'})

if projection == '3d':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if with_clustering:
        ax.scatter(dataset[X_POSITION_COLUMN_NAME], dataset[TIME_COLUMN_NAME], dataset[Y_POSITION_COLUMN_NAME], c=dataset[CLUSTERIZED_COLUMN_NAME], s=1)
    else:
        ax.scatter(dataset[X_POSITION_COLUMN_NAME], dataset[TIME_COLUMN_NAME], dataset[Y_POSITION_COLUMN_NAME], s=1)

    ax.set_xlabel('x[um]')
    ax.set_ylabel('t[s]')
    ax.set_zlabel('y[um]')

    plt.show()

if projection == '2d':
    fig = plt.figure()
    ax = fig.add_subplot()

    if with_clustering:
        ax.scatter(dataset[X_POSITION_COLUMN_NAME], dataset[Y_POSITION_COLUMN_NAME], c=dataset[CLUSTERIZED_COLUMN_NAME], s=1)
    else:
        ax.scatter(dataset[X_POSITION_COLUMN_NAME], dataset[Y_POSITION_COLUMN_NAME], s=1)

    ax.set_xlabel('x[um]')
    ax.set_ylabel('y[um]')

    plt.show()
