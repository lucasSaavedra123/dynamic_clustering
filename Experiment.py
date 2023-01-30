import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import skewnorm

from Cluster import Cluster
from Particle import Particle

from math import ceil

def generate_skewed_normal_distribution(mean, std, skewness, min_value, max_value):
  r = skewnorm.rvs(skewness, loc=mean, scale=std, size=1)[0]

  while not min_value < r < max_value:
    r = skewnorm.rvs(skewness, loc=mean, scale=std, size=1)[0]

  return int(r)

class Experiment():
  def __init__(self,
               height,
               width,
               number_of_clusters_range,
               radio_range,
               number_of_particles_per_cluster_range,
               number_of_initial_non_cluster_particles_range,
               cluster_centroids_diffusion_coefficient_range,
               no_cluster_molecules_diffusion_coefficient_range,
               residence_time_range,
               retention_probabilities_functions_for_each_cluster,
               retention_probabilities,
               lifetime_range,
               lifetime_skewness,
               lifetime_mean,
               lifetime_std,
               eccentricity_maximum,
               minimum_level_of_percentage_molecules,
               average_molecules_per_frame,
               frame_rate,
               plots_with_blinking = False,
               save_memory = True #It is not efficient to hold whole temporal information of the experiment.
               ):

    self.height = height
    self.width = width
    self.max_retention_probability = max(retention_probabilities)
    self.min_retention_probability = min(retention_probabilities)
    self.frame_rate = frame_rate
    self.average_molecules_per_frame = average_molecules_per_frame
    self.cluster_centroids_diffusion_coefficient_range = cluster_centroids_diffusion_coefficient_range
    self.no_cluster_molecules_diffusion_coefficient_range = no_cluster_molecules_diffusion_coefficient_range
    self.plots_with_blinking = plots_with_blinking
    self.time = 0
    self.first_recharge = True
    self.radio_range = radio_range
    self.save_memory = save_memory
    self.residence_time_range = residence_time_range
    self.number_of_particles_per_cluster_range = number_of_particles_per_cluster_range

    self.eccentricity_maximum = eccentricity_maximum

    self.lifetime_range = lifetime_range
    self.lifetime_std = lifetime_std
    self.lifetime_skewness = lifetime_skewness
    self.lifetime_mean = lifetime_mean

    assert 0 < minimum_level_of_percentage_molecules < 1
    self.minimum_level_of_percentage_molecules = minimum_level_of_percentage_molecules
    self.percentage_of_clustered_molecules = None
    self.clustered_molecules = None
    self.non_clustered_molecules = None
    self.going_out_from_cluster_molecules = None
    self.smlm_dataset_rows = []

    self.clusters = []
    self.particles_without_cluster = []
  
    for cluster_index in range(np.random.randint(number_of_clusters_range[0], number_of_clusters_range[1]+1)):
      lifetime = generate_skewed_normal_distribution(self.lifetime_mean, self.lifetime_std, self.lifetime_skewness, self.lifetime_range[0], self.lifetime_range[1])
      self.clusters.append(Cluster(
          np.random.uniform(radio_range[0], radio_range[1]),
          [np.random.uniform(0, width), np.random.uniform(0, height)],
          np.random.randint(number_of_particles_per_cluster_range[0], number_of_particles_per_cluster_range[1]+1),
          np.random.uniform(self.cluster_centroids_diffusion_coefficient_range[0], self.cluster_centroids_diffusion_coefficient_range[1]),
          np.random.choice(retention_probabilities_functions_for_each_cluster, 1)[0],
          lifetime,
          eccentricity_maximum,
          self,
          initial_particles=[]
        )
      )

    #for particle_index in range(np.random.randint(number_of_initial_non_cluster_particles_range[0], number_of_initial_non_cluster_particles_range[1]+1)):
    self.update_percentage_of_clustered_molecules()

    while self.percentage_of_clustered_molecules > self.minimum_level_of_percentage_molecules:
      self.particles_without_cluster.append(
        Particle(
          [np.random.uniform(0, self.width), np.random.uniform(0, self.height)],
          np.random.uniform(no_cluster_molecules_diffusion_coefficient_range[0], no_cluster_molecules_diffusion_coefficient_range[1]),
          self
        )
      )
      self.update_percentage_of_clustered_molecules()

    self.update_percentage_of_clustered_molecules()
    self.recharge_batteries()
    self.scan_for_overlapping_clusters()
    self.update_smlm_dataset()

  def plot(self, t=None, show=False):
    if t is None:
      t = -1

    fig = plt.figure()
    ax = fig.add_subplot()
    particle_size = 1

    clustered_particles = 0
    non_clustered_particles = 0

    for cluster in self.clusters:
      clustered_particles += len(cluster.particles)
      for particle in cluster.particles:
        if not self.plots_with_blinking or particle.blinking_battery != 0:
          ax.scatter(particle.position_at(t)[0], particle.position_at(t)[1], color=particle.color, s=particle_size)

      ax.add_patch(Ellipse( xy=cluster.position_at(t), width=cluster.width, height=cluster.height, angle=cluster.angle*360, fill = False))
      ax.add_patch(plt.Circle( cluster.positions[t,:], cluster.radio , fill = False , linestyle='--'))
      #ax.add_patch(plt.Circle( cluster.positions[t,:], cluster.outer_region.max_radio , fill = False , linestyle='--'))
      #ax.add_patch(plt.Circle( cluster.positions[t,:], cluster.inner_region.min_radio , fill = False , linestyle='--'))
      #ax.add_patch(plt.Circle( cluster.positions[t,:], cluster.center_region.max_radio , fill = False , linestyle='--'))
      #ax.scatter(cluster.positions[t,0], cluster.positions[t,1], marker="X", color="black", s=particle_size)

    non_clustered_particles += len(self.particles_without_cluster)

    for particle in self.particles_without_cluster:
      if not self.plots_with_blinking or particle.blinking_battery != 0:
        ax.scatter(particle.position_at(t)[0], particle.position_at(t)[1], color=particle.color, s=particle_size)

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
    particles_that_dont_belong_no_more_to_cluster = []
    clusters_to_remove = []

    for cluster in self.clusters:
      particles_that_dont_belong_no_more_to_cluster += cluster.move()

      if len(cluster.particles) == 0:
        clusters_to_remove.append(cluster)

    for cluster in clusters_to_remove:
      self.clusters.remove(cluster)

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

    self.update_percentage_of_clustered_molecules()
    self.recharge_batteries()
    self.update_smlm_dataset()
    self.time += 1

  def recharge_batteries(self):

    all_particles = []

    for cluster in self.clusters:
      all_particles += cluster.particles
    
    all_particles += self.particles_without_cluster

    if self.first_recharge == True:
      self.first_recharge = False
      particles_that_will_blink = np.random.choice(all_particles, ceil(self.average_molecules_per_frame), replace=False)

      for particle in particles_that_will_blink:
          particle.blinking_battery = np.random.randint(2, 6)

      self.localizations_that_appear_until_now = 0

    else:
      number_of_particles_currently_blinking = len([particle for particle in all_particles if particle.blinking_battery != 0])
      current_average_molecules_per_frame = (self.localizations_that_appear_until_now+number_of_particles_currently_blinking)/(self.time+1)

      if current_average_molecules_per_frame < self.average_molecules_per_frame:
        number_of_particles_not_currently_blinking = [particle for particle in all_particles if particle.blinking_battery == 0]
        x = abs(ceil(self.average_molecules_per_frame * (self.time+1) - current_average_molecules_per_frame * (self.time+1) - number_of_particles_currently_blinking))
        particles_that_will_blink = np.random.choice(number_of_particles_not_currently_blinking, x, replace=False)

        for particle in particles_that_will_blink:
          particle.blinking_battery = np.random.randint(2, 6)      

      self.localizations_that_appear_until_now += len([particle for particle in all_particles if particle.blinking_battery != 0])
            
  def update_percentage_of_clustered_molecules(self):
    self.clustered_molecules = 0
    self.non_clustered_molecules = 0
    self.going_out_from_cluster_molecules = 0

    for cluster in self.clusters:
      self.clustered_molecules += len(cluster.particles)
    
    self.non_clustered_molecules += len(self.particles_without_cluster)

    for particle in self.particles_without_cluster:
      if particle.going_out_from_cluster:
        self.going_out_from_cluster_molecules += 1

    self.percentage_of_clustered_molecules = self.clustered_molecules/(self.clustered_molecules+self.non_clustered_molecules)

  def one_more_superpass(self):
    return (self.clustered_molecules + self.going_out_from_cluster_molecules + 1)/(self.clustered_molecules+self.non_clustered_molecules) < self.minimum_level_of_percentage_molecules
  
  def update_smlm_dataset(self):
    for particle in self.all_particles():
      if particle.blinking_battery != 0:
        self.smlm_dataset_rows.append({
          'x': particle.position_at(-1)[0],
          'y': particle.position_at(-1)[1],
          't': self.time * self.frame_rate,
          'frame': self.time,
          'clusterized': int(particle.cluster != None),
          'cluster': particle.cluster.id if particle.cluster != None else 0,
        })

  def build_smlm_dataset_as_dataframe(self):
      data = []
      for row in self.smlm_dataset_rows:
          data.append(row)
      return pd.DataFrame(data)

  def all_particles(self):
    all_particles = []

    for cluster in self.clusters:
      all_particles += cluster.particles
    
    all_particles += self.particles_without_cluster

    return all_particles

  def scan_for_overlapping_clusters(self):
    for cluster in self.clusters:
      for other_cluster in self.clusters:
        if cluster != other_cluster:
          if cluster.is_overlapping(other_cluster):
            cluster.move_towards_to(other_cluster)

  def scan_for_merging_clusters(self):
    
    cluster_index = 0
    
    while cluster_index < len(self.clusters):
      for other_cluster in self.clusters:
        if self.clusters[cluster_index] != other_cluster:
          if self.clusters[cluster_index].is_overlapping(other_cluster) and self.cluster[cluster_index].can_merge_with(other_cluster):
            self.clusters.append(Cluster.merge_clusters(self.clusters[cluster_index], other_cluster))
            self.clusters.remove(other_cluster)
            self.clusters.remove(self.clusters[cluster_index])
            cluster_index = 0
            break

      cluster_index += 1

  @property
  def current_time(self):
    return self.time * self.frame_rate
