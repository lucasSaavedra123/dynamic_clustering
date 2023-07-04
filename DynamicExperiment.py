from math import ceil
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import skewnorm

from Cluster import Cluster
from Particle import Particle
from CONSTANTS import *
from utils import custom_norm


def custom_mean(vector):

  number_of_elements = 0
  sum = 0

  for i in vector:
    sum += i
    number_of_elements += 1

  return sum/number_of_elements

def generate_skewed_normal_distribution(mean, std, skewness, min_value, max_value):
  r = skewnorm.rvs(skewness, loc=mean, scale=std, size=1)[0]

  while not min_value < r < max_value:
    r = skewnorm.rvs(skewness, loc=mean, scale=std, size=1)[0]

  return int(r)

class DynamicExperiment():
  def __init__(self,
                height,
                width,
                number_of_clusters_range,
                radio_range,
                number_of_particles_per_cluster_range,
                cluster_centroids_diffusion_coefficient_range,
                no_cluster_molecules_diffusion_coefficient_range,
                residence_time_range,
                anomalous_exponent_range,
                retention_probabilities_functions_for_each_cluster,
                retention_probabilities,
                blinking_consecutives_frames,
                lifetime_range,
                lifetime_skewness,
                lifetime_mean,
                lifetime_std,
                eccentricity_maximum,
                average_molecules_per_frame,
                frame_rate,
                maximum_frame,
                mean_localization_error,
                std_localization_error,
                with_clustering = True,
                with_new_clusters = True,
                max_number_of_no_clusterized_particles = float('inf'),
                number_of_initial_non_cluster_particles_range = None,
                minimum_level_of_percentage_molecules_range = None,
                plots_with_blinking = False,
                save_memory = True #It is not efficient to hold whole temporal information of the experiment.
               ):

    self.with_clustering = with_clustering
    self.with_new_clusters = with_new_clusters
    self.blinking_consecutives_frames = blinking_consecutives_frames
    self.mean_localization_error = mean_localization_error
    self.std_localization_error = std_localization_error
    self.anomalous_exponent_range = anomalous_exponent_range
    self.height = height
    self.width = width
    self.retention_probabilities_functions_for_each_cluster = retention_probabilities_functions_for_each_cluster
    self.max_retention_probability = max(retention_probabilities)
    self.min_retention_probability = min(retention_probabilities)
    self.frame_rate = frame_rate
    self.average_molecules_per_frame = average_molecules_per_frame
    self.cluster_centroids_diffusion_coefficient_range = cluster_centroids_diffusion_coefficient_range
    self.no_cluster_molecules_diffusion_coefficient_range = no_cluster_molecules_diffusion_coefficient_range
    self.plots_with_blinking = plots_with_blinking
    self.time = 0 #This is the current frame
    self.first_recharge = True
    self.radio_range = radio_range
    self.save_memory = save_memory
    self.residence_time_range = residence_time_range
    self.number_of_particles_per_cluster_range = number_of_particles_per_cluster_range

    self.maximum_frame = maximum_frame

    self.eccentricity_maximum = eccentricity_maximum

    self.lifetime_range = lifetime_range
    self.lifetime_std = lifetime_std
    self.lifetime_skewness = lifetime_skewness
    self.lifetime_mean = lifetime_mean

    if minimum_level_of_percentage_molecules_range is None:
      minimum_level_of_percentage_molecules = None

    if minimum_level_of_percentage_molecules_range is not None:
      assert 0 <= minimum_level_of_percentage_molecules_range[0] <= 1
      assert 0 <= minimum_level_of_percentage_molecules_range[1] <= 1
      assert minimum_level_of_percentage_molecules_range[0] <= minimum_level_of_percentage_molecules_range[1]
      minimum_level_of_percentage_molecules = np.random.uniform(minimum_level_of_percentage_molecules_range[0], minimum_level_of_percentage_molecules_range[1])
    elif number_of_initial_non_cluster_particles_range is not None:
      assert number_of_initial_non_cluster_particles_range[0] <= number_of_initial_non_cluster_particles_range[1]
    else:
      assert minimum_level_of_percentage_molecules_range is None and number_of_initial_non_cluster_particles_range is None, "You have to pass a minimum level of percentage molecules or a number of initial non cluster particles range"

    self.minimum_level_of_percentage_molecules = minimum_level_of_percentage_molecules
    self.number_of_initial_non_cluster_particles_range = number_of_initial_non_cluster_particles_range
    self.smlm_dataset_rows = []

    self.clusters = []
    self.particles_without_cluster = []
  
    if self.with_clustering:
      for _ in range(np.random.randint(number_of_clusters_range[0], number_of_clusters_range[1]+1)):
        self.clusters.append(self.generate_cluster_for_experiment())

      if self.minimum_level_of_percentage_molecules is not None:
        while self.percentage_of_clustered_molecules > self.minimum_level_of_percentage_molecules and len(self.particles_without_cluster) < max_number_of_no_clusterized_particles:
          self.particles_without_cluster.append(self.generate_non_clustered_particle_for_experiment())
      elif self.number_of_initial_non_cluster_particles_range is not None:
        number_of_initial_non_cluster_particles = int(np.random.uniform(self.number_of_initial_non_cluster_particles_range[0], self.number_of_initial_non_cluster_particles_range[1]+1))
        for _ in range(number_of_initial_non_cluster_particles):
          self.particles_without_cluster.append(self.generate_non_clustered_particle_for_experiment())
    else:
       if max_number_of_no_clusterized_particles == float('Inf'):
        raise Exception('max_number_of_no_clusterized_particles cannot be infinite if with_clustering is False')
       else:
        self.particles_without_cluster = [self.generate_non_clustered_particle_for_experiment() for _ in range(max_number_of_no_clusterized_particles)]

    self.all_particles = []

    for cluster in self.clusters:
      self.all_particles += cluster.particles
    
    self.all_particles += self.particles_without_cluster

    self.recharge_batteries()
    self.scan_for_merging_clusters()
    self.scan_for_overlapping_clusters()
    self.update_smlm_dataset()

  @property
  def number_of_particles_in_experiment(self):
    return len(self.all_particles)

  def generate_non_clustered_particle_for_experiment(self):
    return Particle(
            [np.random.uniform(0, self.width), np.random.uniform(0, self.height)],
            np.random.uniform(self.no_cluster_molecules_diffusion_coefficient_range[0], self.no_cluster_molecules_diffusion_coefficient_range[1]),
            self
    )

  def generate_cluster_for_experiment(self):
    lifetime = generate_skewed_normal_distribution(self.lifetime_mean, self.lifetime_std, self.lifetime_skewness, self.lifetime_range[0], self.lifetime_range[1])

    return Cluster(
          np.random.uniform(self.radio_range[0], self.radio_range[1]),
          [np.random.uniform(0, self.width), np.random.uniform(0, self.height)],
          np.random.randint(self.number_of_particles_per_cluster_range[0], self.number_of_particles_per_cluster_range[1]+1),
          np.random.uniform(self.cluster_centroids_diffusion_coefficient_range[0], self.cluster_centroids_diffusion_coefficient_range[1]),
          np.random.choice(self.retention_probabilities_functions_for_each_cluster, 1)[0],
          lifetime,
          self.eccentricity_maximum,
          self,
          initial_particles=[]
    )

  def plot(self, t=None, show=False):
    if t is None:
      t = -1

    fig = plt.figure()
    ax = fig.add_subplot()
    particle_size = 1
    clustered_particles = []

    for cluster in self.clusters:
      ax.add_patch(Ellipse( xy=cluster.position_at(t), width=cluster.width, height=cluster.height, angle=cluster.angle*360, fill = False))
      clustered_particles += cluster.particles
      #ax.add_patch(plt.Circle( cluster.positions[t,:], cluster.radio , fill = False , linestyle='--'))
      #ax.add_patch(plt.Circle( cluster.positions[t,:], cluster.outer_region.max_radio , fill = False , linestyle='--'))
      #ax.add_patch(plt.Circle( cluster.positions[t,:], cluster.inner_region.min_radio , fill = False , linestyle='--'))
      #ax.add_patch(plt.Circle( cluster.positions[t,:], cluster.center_region.max_radio , fill = False , linestyle='--'))
      #ax.scatter(cluster.positions[t,0], cluster.positions[t,1], marker="X", color="black", s=particle_size)

    clustered_particles_as_array = np.array([particle.position_at(t) for particle in clustered_particles if not self.plots_with_blinking or particle.blinking_battery != 0])

    if len(clustered_particles_as_array) > 0:
      ax.scatter(clustered_particles_as_array[:,0], clustered_particles_as_array[:,1], color=[a_particle.color for a_particle in clustered_particles], s=particle_size)

    non_clustered_particles_as_array = np.array([particle.position_at(t) for particle in self.particles_without_cluster if not self.plots_with_blinking or particle.blinking_battery != 0])

    if len(non_clustered_particles_as_array) > 0:
      ax.scatter(non_clustered_particles_as_array[:,0], non_clustered_particles_as_array[:,1], color=[a_particle.color for a_particle in self.particles_without_cluster], s=particle_size)

    ax.set_aspect('equal')
    ax.set_title(f'Clustered Particles:{round(self.percentage_of_clustered_molecules*100, 2)}%, t={self.time*10}ms')
    ax.set_xlim([0, self.width])
    ax.set_ylim([0, self.height])
    ax.set_xlabel('x [um]')
    ax.set_ylabel('y [um]')

    #ax.set_xlim([cluster.position_at(t)[0]-cluster.radio, cluster.position_at(t)[0]+cluster.radio])
    #ax.set_ylim([cluster.position_at(t)[1]-cluster.radio, cluster.position_at(t)[1]+cluster.radio])

    if show:
      plt.show()

  def move(self):
    self.time += 1
    assert self.time != self.maximum_frame, "The simulation has already finished (the limit is maximum_frames parameter)"
    particles_that_dont_belong_no_more_to_cluster = []
    clusters_to_remove = []

    for cluster in self.clusters:
      particles_that_dont_belong_no_more_to_cluster += cluster.move()

      if len(cluster.particles) == 0 or all([particle_in_cluster.locked for particle_in_cluster in cluster.particles]):
        clusters_to_remove.append(cluster)

        for particle in cluster.particles:
          particle.locked = np.random.choice([False, True])
          particle.cluster = None
          particle.diffusion_coefficient = np.random.uniform(self.no_cluster_molecules_diffusion_coefficient_range[0], self.no_cluster_molecules_diffusion_coefficient_range[1])
          particles_that_dont_belong_no_more_to_cluster.append(particle)

    for cluster in clusters_to_remove:
      self.clusters.remove(cluster)
      cluster.exist = False

    for particle in self.particles_without_cluster:
      particle.move()

    new_particles_without_cluster = []

    for particle in self.particles_without_cluster + particles_that_dont_belong_no_more_to_cluster:
        cluster_assigned = False
        for cluster in self.clusters:
            if cluster.is_inside(particle):
                cluster.add_particle(particle)
                cluster_assigned = True
                break

        if not cluster_assigned:
          new_particles_without_cluster.append(particle)

    self.particles_without_cluster = new_particles_without_cluster

    if self.with_clustering and self.with_new_clusters:
      new_clusters = self.scan_for_new_clusters()
    else:
      new_clusters = []

    for cluster in new_clusters:
        new_particles = [a_particle for a_particle in self.particles_without_cluster if cluster.is_inside(a_particle)]
        for new_particle in new_particles:
            self.particles_without_cluster.remove(new_particle)
            cluster.add_particle(new_particle)

    assert self.number_of_particles_in_experiment == sum([len(cluster.particles) for cluster in self.clusters]) + len(self.particles_without_cluster), "Particles dissapeared during simulation"

    self.scan_for_merging_clusters()
    self.scan_for_overlapping_clusters()
    self.recharge_batteries()
    self.update_smlm_dataset()

  def recharge_batteries(self):
    all_particles = [particle for particle in self.all_particles if particle.in_fov()]

    if self.first_recharge == True:
      self.first_recharge = False
      particles_that_will_blink = np.random.choice(all_particles, ceil(self.average_molecules_per_frame), replace=False)

      for particle in particles_that_will_blink:
          particle.blinking_battery = np.random.randint(self.blinking_consecutives_frames[0], self.blinking_consecutives_frames[1]+1)

      self.localizations_that_appear_until_now = 0

    else:
      number_of_particles_currently_blinking = len([particle for particle in all_particles if particle.blinking_battery != 0])
      current_average_molecules_per_frame = (self.localizations_that_appear_until_now+number_of_particles_currently_blinking)/(self.time+1)

      if current_average_molecules_per_frame < self.average_molecules_per_frame:
        number_of_particles_not_currently_blinking = [particle for particle in all_particles if particle.blinking_battery == 0]
        x = abs(ceil(self.average_molecules_per_frame * (self.time+1) - current_average_molecules_per_frame * (self.time+1) - number_of_particles_currently_blinking))
        particles_that_will_blink = np.random.choice(number_of_particles_not_currently_blinking, x, replace=False)

        for particle in particles_that_will_blink:
          particle.blinking_battery = np.random.randint(self.blinking_consecutives_frames[0], self.blinking_consecutives_frames[1]+1)

      self.localizations_that_appear_until_now += len([particle for particle in all_particles if particle.blinking_battery != 0])

  @property    
  def percentage_of_clustered_molecules(self): 
    clustered_molecules = 0
    non_clustered_molecules = 0

    for cluster in self.clusters:
      clustered_molecules += len(cluster.particles)

    non_clustered_molecules += len(self.particles_without_cluster)

    if clustered_molecules == 0:
      return 0
    else:
      return clustered_molecules/(clustered_molecules+non_clustered_molecules)

  def generate_noise(self):
    error = np.random.normal(loc=self.mean_localization_error / 2, scale=self.std_localization_error / 2, size=1)[0]
    error_sign = np.random.choice([-1, 1], size=1)[0]
    return error * error_sign

  def update_smlm_dataset(self):
    for particle in [a_particle for a_particle in self.all_particles if a_particle.in_fov() and a_particle.blinking_battery != 0]:
      self.smlm_dataset_rows.append({
        PARTICLE_ID_COLUMN_NAME: particle.id,
        X_POSITION_COLUMN_NAME: particle.position_at(-1)[0] + self.generate_noise(),
        Y_POSITION_COLUMN_NAME: particle.position_at(-1)[1] + self.generate_noise(),
        TIME_COLUMN_NAME: self.current_time,
        FRAME_COLUMN_NAME: self.time,
        CLUSTERIZED_COLUMN_NAME: int(particle.cluster != None),
        CLUSTER_ID_COLUMN_NAME: particle.cluster.id if particle.cluster != None else 0,
      })

  def build_smlm_dataset_as_dataframe(self):
      return pd.DataFrame(self.smlm_dataset_rows)

  def scan_for_overlapping_clusters(self):
    for cluster in self.clusters:
      for other_cluster in self.clusters:
        if cluster.cluster_moving_to is None and other_cluster.cluster_moving_to is None and cluster != other_cluster and cluster.is_overlapping(other_cluster):
          cluster.move_towards_to(other_cluster)

  def scan_for_merging_clusters(self):
    cluster_index = 0
    
    while cluster_index < len(self.clusters):
      for other_cluster in self.clusters:
        if self.clusters[cluster_index] != other_cluster:
          if self.clusters[cluster_index].can_merge_with(other_cluster):
            self.clusters.append(Cluster.merge_clusters(self.clusters[cluster_index], other_cluster))
            self.clusters.remove(other_cluster)
            self.clusters.remove(self.clusters[cluster_index])
            cluster_index = 0
            break

      cluster_index += 1

  def scan_for_new_clusters(self):
    non_clustered_molecule_index = 0
    non_clustered_molecules = [particle for particle in self.all_particles if particle.cluster is None and not particle.came_from_existent_cluster()]

    new_clusters = []

    candidate_new_cluster = Cluster(
        np.random.uniform(self.radio_range[0], self.radio_range[1]),
        [self.width/2, self.height/2],
        0,
        np.random.uniform(self.cluster_centroids_diffusion_coefficient_range[0], self.cluster_centroids_diffusion_coefficient_range[1]),
        np.random.choice(self.retention_probabilities_functions_for_each_cluster, 1)[0],
        generate_skewed_normal_distribution(self.lifetime_mean, self.lifetime_std, self.lifetime_skewness, self.lifetime_range[0], self.lifetime_range[1]),
        self.eccentricity_maximum,
        self,
        initial_particles=[]
      )

    """
    Distance between particles is symmetric: d(p_1, p_2) = d(p_2, p_1)
    """

    distance_dictionary = {}

    for molecule_one_index in range(len(non_clustered_molecules)):
      molecule_one = non_clustered_molecules[molecule_one_index]
      distance_dictionary[molecule_one] = {}

      for molecule_two_index in range(molecule_one_index, len(non_clustered_molecules)):
        molecule_two = non_clustered_molecules[molecule_two_index]
        distance_dictionary[molecule_one][molecule_two] = custom_norm(molecule_one.position_at(-1), molecule_two.position_at(-1))

    def get_distance_between_molecules(molecule_one, molecule_two):
      if molecule_two in distance_dictionary[molecule_one]:
        return distance_dictionary[molecule_one][molecule_two]
      else:
        return distance_dictionary[molecule_two][molecule_one]

    while non_clustered_molecule_index < len(non_clustered_molecules):
      particle = non_clustered_molecules[non_clustered_molecule_index]

      """
      We used to call np.linalg.norm to measure the distance between points:

      particles_sorted_by_distance = sorted(
        non_clustered_molecules,
        key=lambda x: np.linalg.norm(particle.position_at(-1) - x.position_at(-1))
      )

      After a profiling analysis done to the simulation, we found out that 
      np.linalg.norm was quite slow. Then, we were wondering if this was something
      related with the numpy implementation or was a CPU-related problem.

      To answer this, we implemented the function 'custom_norm'. The times changed
      drastically:

      One move with np.linalg.norm: 54.602 seconds
      One move with custom_norm: 22.344 seconds

      Apparently, np.linalg.norm has little overhead for little arrays
      """

      particles_sorted_by_distance = sorted(non_clustered_molecules, key=lambda x: get_distance_between_molecules(particle, x))
      positions_of_all_particles_in_system = np.array([[]])
      list_of_new_particles = []
      old_centroid = None

      for neighbor_particle in particles_sorted_by_distance:
        if len(list_of_new_particles) == 0:
          positions_of_all_particles_in_system = np.append(positions_of_all_particles_in_system, np.array([neighbor_particle.position_at(-1)]), axis=1)
        else:
          positions_of_all_particles_in_system = np.append(positions_of_all_particles_in_system, np.array([neighbor_particle.position_at(-1)]), axis=0)

        candidate_new_cluster.positions = np.array([[custom_mean(positions_of_all_particles_in_system[:, 0]), custom_mean(positions_of_all_particles_in_system[:, 1])]])

        if all([candidate_new_cluster.is_inside(particle_aux) for particle_aux in list_of_new_particles+[neighbor_particle]]):
          list_of_new_particles.append(neighbor_particle)
          old_centroid = candidate_new_cluster.positions
        else:
          break

      candidate_new_cluster.positions = old_centroid

      if len(list_of_new_particles) >= min(self.number_of_particles_per_cluster_range):
        for new_clustered_particle in list_of_new_particles:
          non_clustered_molecules.remove(new_clustered_particle)

        self.clusters.append(candidate_new_cluster)
        new_clusters.append(candidate_new_cluster)
        non_clustered_molecule_index = 0

        candidate_new_cluster = Cluster(
            np.random.uniform(self.radio_range[0], self.radio_range[1]),
            [self.width/2, self.height/2],
            0,
            np.random.uniform(self.cluster_centroids_diffusion_coefficient_range[0], self.cluster_centroids_diffusion_coefficient_range[1]),
            np.random.choice(self.retention_probabilities_functions_for_each_cluster, 1)[0],
            generate_skewed_normal_distribution(self.lifetime_mean, self.lifetime_std, self.lifetime_skewness, self.lifetime_range[0], self.lifetime_range[1]),
            self.eccentricity_maximum,
            self,
            initial_particles=[]
          )

      else:
        non_clustered_molecule_index += 1

    return new_clusters

  @property
  def current_time(self):
    return self.time * self.frame_rate

  def save_plot(self, path="./", dpi=200):
    self.plot(show=False)
    plt.savefig(os.path.join(path, f"{str(self.time).zfill(10)}.jpg"), dpi=dpi)
    plt.close()

  @property
  def summary_as_string(self):
    string_text = ""
    for attribute in self.__dict__:
      if attribute in ['smlm_dataset_rows', 'clusters', 'particles_without_cluster', 'all_particles']:
        string_text += f"{attribute}: {len(self.__dict__[attribute])}\n"
      else:
        string_text += f"{attribute}: {self.__dict__[attribute]}\n"    

    string_text += f"percentage_of_clustered_molecules: {self.percentage_of_clustered_molecules}\n"

    return string_text

  def summary(self):
    print("All attributes:")
    print(self.summary_as_string)

  def save_summary(self, path="./specs.txt"):
    print(f"Saving attributes in {path}")
    with open(path, "w") as file:
      file.write(self.summary_as_string)