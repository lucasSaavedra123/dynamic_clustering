import sys

import numpy as np
import matplotlib.pyplot as plt

from Experiment import Experiment


an_experiment = Experiment(
    10, #um
    10, #um
    [10, 100], #number_of_clusters_range
    [0.02, 0.2], #radio_range
    [15, 15], #number_of_particles_per_cluster_range
    [100, 100], #number_of_initial_non_cluster_particles_range
    [1e-5, 0.01], #cluster_centroids_diffusion_coefficient_range
    [1e-5, 0.7], #no_cluster_molecules_diffusion_coefficient_range
    [0.358, 0.025],
    [25, 7000],
    3,
    500,
    2000,
    0.7,
    0.85,
    1,
    10e-3, #frame_rate
    plots_with_blinking = False,
    save_memory = True
)

print("Primera foto...")
an_experiment.plot(show=False)
plt.savefig(f"./images/{str(0).zfill(5)}.jpg", dpi=200)
plt.clf()

clustered_molecules = []
non_clustered_molecules = []

for i in range(1, 100):
    print("Step:", i)
    an_experiment.move()
    print(an_experiment.percentage_of_clustered_molecules)
    print("Ploting...", i)
    an_experiment.plot(show=False)
    plt.savefig(f"./images/{str(i).zfill(5)}.jpg", dpi=200)
    plt.clf()
    clustered_molecules += [an_experiment.clustered_molecules]
    non_clustered_molecules += [an_experiment.non_clustered_molecules]

"""
plt.plot(clustered_molecules, label="Clustered molecules", color="red")
plt.plot(non_clustered_molecules, label="Non clustered molecules", color="black")
plt.grid()
plt.legend()
plt.xlabel("Time (frame)")
plt.xlabel("Number of molecules")
plt.show()
"""

#an_experiment.build_smlm_dataset_as_dataframe().to_csv("smlm_dataset.csv", index=False)
