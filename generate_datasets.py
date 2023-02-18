import numpy as np

from Experiment import Experiment
from RetentionProbabilities import *

"""
THIS INCLUDES SIMULATION PARAMETERS DIFFERENT FROM ORIGINAL BECAUSE,
FIRST, WE WANT TO FIGURE OUT HOW GNNs ARE IMPLEMENTED AND TO HAVE SOME
DATASETS TO INITIATE IN THIS FIELD.
"""

number = 0

while True:
    average_localizations_per_frame = np.random.uniform(10, 50)

    an_experiment = Experiment(
        10, #um
        10, #um
        [10, 100], #number_of_clusters_range
        [0.02, 0.2], #radio_range
        [10, 100], #number_of_particles_per_cluster_range
        [1e-5, 0.01], #cluster_centroids_diffusion_coefficient_range
        [1e-5, 0.7], #no_cluster_molecules_diffusion_coefficient_range
        [0.358, 0.025],
        [0.1, 1.9],
        [RetentionProbabilityWithDiscreteFunction, RetentionProbabilityWithCuadraticFunction, RetentionProbabilityWithLinearFunction],
        [0.01, 0.5],
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
        #number_of_initial_non_cluster_particles_range = [2000, 2000], #number_of_initial_non_cluster_particles_range
        minimum_level_of_percentage_molecules_range = [0, 1],
        plots_with_blinking = False,
        save_memory = True
    )

    an_experiment.summary()
    an_experiment.save_plot(path=f"./images/{number+1}")

    for i in range(0, 999):
        an_experiment.move()
        an_experiment.save_plot(path=f"./images/{number+1}")
    
    an_experiment.build_smlm_dataset_as_dataframe().to_csv(f"./images/{number+1}/smlm_dataset.csv", index=False)
    number += 1
