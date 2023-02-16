from Experiment import Experiment
from RetentionProbabilities import *


if __name__ == '__main__':
    for number in range(1, 8+1):
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
            3.6,
            10e-3, #frame_rate
            7000,
            40/1000,
            10/1000,
            #number_of_initial_non_cluster_particles_range = [2000, 2000], #number_of_initial_non_cluster_particles_range
            minimum_level_of_percentage_molecules_range = [0, 0.85],
            plots_with_blinking = False,
            save_memory = True
        )

        for i in range(0, 1000):
            print(number, i)
            an_experiment.move()
        
        an_experiment.build_smlm_dataset_as_dataframe().to_csv(f"datasets/smlm_dataset_{number}.csv", index=False)
