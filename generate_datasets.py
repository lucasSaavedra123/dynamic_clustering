import numpy as np

from Experiment import Experiment, ExperimentException
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
    try:
        resimulate = True
        while resimulate:
            average_localizations_per_frame = np.random.uniform(10, 100)

            """
            References:

            [1] Mosqueira, A., Camino, P. A., & Barrantes, F. J. (2018). 
                Cholesterol modulates acetylcholine receptor diffusion by tuning confinement sojourns and nanocluster stability. 
                Scientific Reports, 8(1). doi:10.1038/s41598-018-30384-y
            
            [2] Mosqueira, A., Camino, P. A., & Barrantes, F. J. (2020).
                Antibody-induced crosslinking and cholesterol-sensitive, anomalous diffusion of nicotinic acetylcholine receptors.
                J Neurochem, 152(6), 663-674. doi:10.1111/jnc.14905
            """

            """
            The designed sorftware is unit agnostic. So, be careful with the used magnitudes.
            """

            an_experiment = Experiment(
                10, #um
                10, #um
                [0, 100], #number_of_clusters_range
                [0.02, 0.2], #radio_range [1], [2]
                [10, 100], #number_of_particles_per_cluster_range [1]
                [1e-5, 0.01], #cluster_centroids_diffusion_coefficient_range
                [1e-5, 0.7], #no_cluster_molecules_diffusion_coefficient_range
                [0.358, 0.025], #residence_time_range
                [0.1, 1.9], #anomalous_exponent_range
                [RetentionProbabilityWithDiscreteFunction, RetentionProbabilityWithCuadraticFunction, RetentionProbabilityWithLinearFunction],
                [0.01, 0.5], #retention_probabilities
                [2, 5],
                [25, 7000],
                3,
                500,
                2000,
                0.6, #eccentricity_maximum [1]
                average_localizations_per_frame,
                10e-3, #frame_rate
                7000, #maximum_frame
                40/1000, #std_localization_error
                10/1000, #mean_localization_error
                with_clustering=True,
                with_new_clusters=True,
                max_number_of_no_clusterized_particles= 10000,
                minimum_level_of_percentage_molecules_range = [0, 1],
                plots_with_blinking = False,
                save_memory = True
            )

            images_path = f'{file_number}_images'

            if not os.path.exists(images_path):
                os.mkdir(images_path)

            an_experiment.save_summary(path=os.path.join(directory_path, f"{file_number}_specs.txt"))
            an_experiment.save_plot(images_path)
            print(file_number, 0)

            for i in range(1, 1000):
                print(file_number, i)
                an_experiment.move()
                an_experiment.save_plot(images_path)

            an_experiment.build_smlm_dataset_as_dataframe().to_csv(os.path.join(directory_path, f"{file_number}_{file_suffix}.csv"), index=False)
            resimulate=False
    except ExperimentException:
        print(f"Experiment {file_number} needs resimulation...")
        resimulate=True
