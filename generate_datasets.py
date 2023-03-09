import numpy as np

from Experiment import Experiment
from RetentionProbabilities import *

from argparse import ArgumentParser
import os

"""
THIS INCLUDES SIMULATION PARAMETERS DIFFERENT FROM ORIGINAL BECAUSE,
FIRST, WE WANT TO FIGURE OUT HOW GNNs ARE IMPLEMENTED AND TO HAVE SOME
DATASETS TO INITIATE IN THIS FIELD.
"""
parser = ArgumentParser()
parser.add_argument("-d", "--directory", dest="directory", default="./datasets")
args = parser.parse_args()

file_suffix = 'smlm_dataset'
directory_path = os.path.join('./', args.directory)

def next_dataset_number_generator():
    global file_suffix, directory_path

    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)

    number = 0

    while True:
        if not os.path.isfile(os.path.join(directory_path, f"{number}_{file_suffix}.csv")):
            yield number
        number += 1

file_number_generator = next_dataset_number_generator()

for file_number in file_number_generator:
    average_localizations_per_frame = np.random.uniform(10, 50)

    an_experiment = Experiment(
        10, #um
        10, #um
        [0, 100], #number_of_clusters_range
        [0.02, 0.2], #radio_range
        [10, 100], #number_of_particles_per_cluster_range
        [1e-5, 0.01], #cluster_centroids_diffusion_coefficient_range
        [1e-5, 0.7], #no_cluster_molecules_diffusion_coefficient_range
        [0.358, 0.025],
        [0.1, 1.9],
        [RetentionProbabilityWithDiscreteFunction, RetentionProbabilityWithCuadraticFunction, RetentionProbabilityWithLinearFunction],
        [0.01, 0.5],
        [2, 5],
        [25, 7000],
        3,
        500,
        2000,
        0.6,
        average_localizations_per_frame,
        10e-3, #frame_rate
        7000,
        40/1000,
        10/1000,
        with_clustering=True,
        with_new_clusters=True,
        max_number_of_no_clusterized_particles= 10000,
        minimum_level_of_percentage_molecules_range = [0, 1],
        plots_with_blinking = False,
        save_memory = True
    )

    an_experiment.save_summary(path=os.path.join(directory_path, f"{file_number}_specs.txt"))

    for i in range(0, 999):
        print(file_number, i)
        an_experiment.move()

    an_experiment.build_smlm_dataset_as_dataframe().to_csv(os.path.join(directory_path, f"{file_number}_{file_suffix}.csv"), index=False)
