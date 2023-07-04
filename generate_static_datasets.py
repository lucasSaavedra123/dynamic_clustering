import numpy as np

from StaticExperiment import StaticExperiment
from RetentionProbabilities import RetentionProbabilityEmpty

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

directory_path = os.path.join('./', args.directory)

if not os.path.isdir(directory_path):
    os.mkdir(directory_path)

total_txt_files = len([file for file in os.listdir(directory_path) if file.endswith('.txt')])
total_csv_files = len([file for file in os.listdir(directory_path) if file.endswith('.csv')])

if total_txt_files != total_csv_files:
    number = total_txt_files - 1
else:
    number = total_txt_files

while True:

    an_experiment = StaticExperiment(
        10, #um
        10, #um
        [0, 100], #number_of_clusters_range
        [0.02, 0.2], #radio_range
        [10, 100], #number_of_particles_per_cluster_range
        0.6,
        40/1000,
        10/1000,
        with_clustering=np.random.choice([True, False]),
        max_number_of_no_clusterized_particles= 10000,
        minimum_level_of_percentage_molecules_range = [0, 1],
    )

    an_experiment.save_summary(path=os.path.join(directory_path, f"{number}_specs.txt"))
    an_experiment.update_smlm_dataset()
    an_experiment.build_smlm_dataset_as_dataframe().to_csv(os.path.join(directory_path, f"{number}_smlm_dataset.csv"), index=False)
    number += 1
