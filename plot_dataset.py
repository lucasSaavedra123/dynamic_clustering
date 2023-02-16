import pandas as pd
import matplotlib.pyplot as plt

from CONSTANTS import *

dataset = pd.read_csv('smlm_dataset_4.csv')

print(f"Average: {len(dataset)/max(dataset[FRAME_COLUMN_NAME])}")

projection = '2d'
with_clustering = False

if with_clustering:
    dataset[CLUSTERIZED_COLUMN_NAME] = dataset[CLUSTERIZED_COLUMN_NAME].map({0: 'black', 1: 'red'})

if projection == '3d':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if with_clustering:
        ax.scatter(dataset[X_POSITION_COLUMN_NAME], dataset[TIME_COLUMN_NAME], dataset[Y_POSITION_COLUMN_NAME], c=dataset[CLUSTERIZED_COLUMN_NAME], s=1)
    else:
        ax.scatter(dataset[X_POSITION_COLUMN_NAME], dataset[TIME_COLUMN_NAME], dataset[Y_POSITION_COLUMN_NAME], s=1)

    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('y')

    plt.show()

if projection == '2d':
    fig = plt.figure()
    ax = fig.add_subplot()

    if with_clustering:
        ax.scatter(dataset[X_POSITION_COLUMN_NAME], dataset[Y_POSITION_COLUMN_NAME], c=dataset[CLUSTERIZED_COLUMN_NAME], s=1)
    else:
        ax.scatter(dataset[X_POSITION_COLUMN_NAME], dataset[Y_POSITION_COLUMN_NAME], s=1)

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.show()
