from math import ceil
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import skewnorm

from Cluster import Cluster
from RetentionProbabilities import RetentionProbabilityEmpty
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

class StaticExperiment():
  def __init__(self,
                height,
                width,
                number_of_clusters_range,
                radio_range,
                number_of_particles_per_cluster_range,
                eccentricity_maximum,
                mean_localization_error,
                std_localization_error,
                with_clustering = True,
                max_number_of_no_clusterized_particles = float('inf'),
                number_of_initial_non_cluster_particles_range = None,
                minimum_level_of_percentage_molecules_range = None,
               ):

    self.with_clustering = with_clustering
    self.mean_localization_error = mean_localization_error
    self.std_localization_error = std_localization_error
    self.height = height
    self.width = width
    self.radio_range = radio_range
    self.number_of_particles_per_cluster_range = number_of_particles_per_cluster_range

    self.residence_time_range = [1,2]
    self.anomalous_exponent_range = [1,1]
    self.maximum_frame = 1

    self.eccentricity_maximum = eccentricity_maximum

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

    self.scan_for_overlapping_clusters()

  @property
  def current_time(self):
    return 0

  @property
  def number_of_particles_in_experiment(self):
    return len(self.all_particles)

  def generate_non_clustered_particle_for_experiment(self):
    retry = True

    while retry:
      generated_particle = Particle(
            [np.random.uniform(0, self.width), np.random.uniform(0, self.height)],
            np.random.uniform(0, 0),
            self
      )

      retry = any([a_cluster.is_inside(particle=generated_particle) for a_cluster in self.clusters])

    return generated_particle

  def generate_cluster_for_experiment(self):

    return Cluster(
          np.random.uniform(self.radio_range[0], self.radio_range[1]),
          [np.random.uniform(0, self.width), np.random.uniform(0, self.height)],
          np.random.randint(self.number_of_particles_per_cluster_range[0], self.number_of_particles_per_cluster_range[1]+1),
          np.random.uniform(0, 0),
          RetentionProbabilityEmpty,
          float('Inf'),
          self.eccentricity_maximum,
          self,
          initial_particles=[]
    )

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
    for particle in [a_particle for a_particle in self.all_particles if a_particle.in_fov()]:
      self.smlm_dataset_rows.append({
        PARTICLE_ID_COLUMN_NAME: particle.id,
        X_POSITION_COLUMN_NAME: particle.position_at(-1)[0] + self.generate_noise(),
        Y_POSITION_COLUMN_NAME: particle.position_at(-1)[1] + self.generate_noise(),
        TIME_COLUMN_NAME: 0,
        FRAME_COLUMN_NAME: 0,
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
