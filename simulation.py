import sys

import numpy as np
import matplotlib.pyplot as plt

from Experiment import Experiment

from andi_datasets import analysis

from RetentionProbabilities import *

an_experiment = Experiment(
    10, #um
    10, #um
    [20, 20], #number_of_clusters_range
    [0.02, 0.2], #radio_range
    [15, 15], #number_of_particles_per_cluster_range
    [100, 100], #number_of_initial_non_cluster_particles_range
    [1e-5, 0.01], #cluster_centroids_diffusion_coefficient_range
    [1e-5, 0.7], #no_cluster_molecules_diffusion_coefficient_range
    [0.358, 0.025],
    [RetentionProbabilityWithDiscreteFunction, RetentionProbabilityWithCuadraticFunction, RetentionProbabilityWithLinearFunction],
    [0.01, 0.5],
    [25, 7000],
    3,
    500,
    2000,
    0.7,
    1,
    3.6,
    10e-3, #frame_rate
    500,
    plots_with_blinking = False,
    save_memory = True
)

an_experiment.plot(show=False)
plt.savefig(f"./images/{str(0).zfill(5)}.jpg", dpi=200)
plt.clf()

for i in range(1, 500):
    print("Step:", i)
    an_experiment.move()
    #print(an_experiment.percentage_of_clustered_molecules)
    print("Ploting...", i)
    an_experiment.plot(show=False)
    plt.savefig(f"./images/{str(i).zfill(5)}.jpg", dpi=200)
    plt.clf()

#an_experiment.build_smlm_dataset_as_dataframe().to_csv("smlm_dataset.csv", index=False)
