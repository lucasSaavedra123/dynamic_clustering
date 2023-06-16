import itertools

import numpy as np

#from hypo import Hypoexponential
from Particle import Particle

from utils import custom_norm

class Cluster():
  id_obj = itertools.count(1)

  @classmethod
  def merge_clusters(cls, cluster1, cluster2):
    if cluster1.is_inside_of_cluster(cluster2):
      new_position = cluster2.position_at(-1)
    elif cluster2.is_inside_of_cluster(cluster1):
      new_position = cluster1.position_at(-1)
    else:
      new_position = (cluster1.position_at(-1) + cluster2.position_at(-1))/2

    return Cluster(
      (cluster1.radio + cluster2.radio) / 2,
      new_position,
      0,
      (cluster1.centroid_diffusion_coefficient + cluster2.centroid_diffusion_coefficient) / 2,
      np.random.choice([cluster1.retention_probability_function.__class__, cluster2.retention_probability_function.__class__], 1)[0],
      np.random.choice([cluster1.lifetime, cluster2.lifetime], 1)[0],
      np.random.choice([cluster1.eccentricity_maximum, cluster2.eccentricity_maximum], 1)[0],
      cluster1.experiment,
      initial_particles = cluster1.particles+cluster2.particles
    )

  @property
  def eccentricity(self):
    if self.height < self.width:
      a = self.height
      b = self.width
    else:
      a = self.width
      b = self.height

    return np.sqrt(1 - (a**2/b**2))

  def is_inside(self, particle=None, position=None):
    if particle is not None:
      x = particle.position_at(-1)[0]
      y = particle.position_at(-1)[1]
    elif position is not None:
      x = position[0]
      y = position[1]

    # The ellipse
    g_ell_center = self.position_at(-1)
    g_ell_width = self.width
    g_ell_height = self.height
    angle = self.angle

    #cos_angle = np.cos(np.radians(180.-angle))
    #sin_angle = np.sin(np.radians(180.-angle))

    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    xc = x - g_ell_center[0]
    yc = y - g_ell_center[1]

    xct = xc * cos_angle - yc * sin_angle
    yct = xc * sin_angle + yc * cos_angle 

    return (xct**2/(g_ell_width/2.)**2) + (yct**2/(g_ell_height/2.)**2) <= 1

  def distance_to_radio_from(self, position, t=-1):
    return custom_norm(position, self.position_at(t))

  def add_particle(self, particle):
    self.particles.append(particle)
    particle.cluster = self
    factor = np.random.uniform(self.min_factor, self.max_factor)
    particle.diffusion_coefficient = self.centroid_diffusion_coefficient/factor
    particle.can_be_retained = np.random.choice([False, True], 1, p=[0.90, 0.10])[0]
    #particle.residence_time = Hypoexponential(self.experiment.residence_time_range).sample(1)[0]
    particle.residence_time = np.random.uniform(self.experiment.residence_time_range[0], self.experiment.residence_time_range[1])
    particle.going_out_from_cluster = False
    particle.time_belonging_cluster = self.experiment.current_time

  def __init__(self, radio, initial_position, number_of_initial_particles, centroid_diffusion_coefficient, retention_probability_function, lifetime, eccentricity_maximum, experiment, initial_particles=[]):
    self.radio = radio
    self.number_of_particles_leaving_cluster = 0
    self.positions = np.array([initial_position])
    self.particles = []
    self.experiment = experiment
    self.number_of_particles_going_out = 0
    self.eccentricity_maximum = eccentricity_maximum
    self.retention_probability_function = retention_probability_function() # We create an instance of it
    self.id = next(Cluster.id_obj)
    self.exist = True
    self.lifetime = lifetime
    self.centroid_diffusion_coefficient = centroid_diffusion_coefficient
    self.cluster_moving_to = None
    bad_initial_shape = True

    self.slower_than_particles = np.random.choice([False, True], 1, p=[0.5, 0.5])[0]

    if self.slower_than_particles:
      self.min_factor = 0.5
      self.max_factor = 1
    else:
      self.min_factor = 1
      self.max_factor = 2

    while bad_initial_shape:
      self.width = np.random.uniform(radio, self.experiment.radio_range[1]) * 2
      self.height = np.random.uniform(radio, self.experiment.radio_range[1]) * 2
      self.angle = np.random.uniform(0, 2*np.pi)

      if self.eccentricity > self.eccentricity_maximum:
        bad_initial_shape = True
      else:
        bad_initial_shape = False

    if initial_particles != []:
      for particle in initial_particles:
        self.add_particle(particle)
    else:
      for _ in range(number_of_initial_particles):
        not_inside = True

        while not_inside:
          angle = np.random.uniform(0, 2 * np.pi)
          radio = np.random.uniform(0, max(self.height/2, self.width/2))
          new_position = [initial_position[0] + np.cos(angle) * radio, initial_position[1] + np.sin(angle) * radio]
          not_inside = not self.is_inside(position=new_position)

        self.particles.append(self.generate_particle_for_cluster(new_position))

  def generate_particle_for_cluster(self, position):
    return Particle(
      position,
      self.centroid_diffusion_coefficient / np.random.uniform(self.min_factor, self.max_factor),
      self.experiment,
      can_be_retained=np.random.choice([False, True], 1, p=[0.95, 0.05]),
      cluster=self,
      #residence_time=Hypoexponential(self.experiment.residence_time_range).sample(1)[0]
      residence_time = np.random.uniform(self.experiment.residence_time_range[0], self.experiment.residence_time_range[1])
    )

  def probability_to_be_retained(self, particle):
    assert particle in self.particles
    return self.retention_probability_function(particle)

  def move(self):
    particles_without_cluster = []

    new_x = self.position_at(-1)[0] + np.sqrt(2*self.centroid_diffusion_coefficient*self.experiment.frame_rate) * np.random.normal(0,1)
    new_y = self.position_at(-1)[1] + np.sqrt(2*self.centroid_diffusion_coefficient*self.experiment.frame_rate) * np.random.normal(0,1)

    if self.cluster_moving_to is not None:
      direction_to_another_cluster = self.cluster_moving_to.position_at(-1) - self.position_at(-1)
      new_x = self.position_at(-1)[0] + np.sign(direction_to_another_cluster[0]) * np.abs(np.sqrt(2*self.centroid_diffusion_coefficient*self.experiment.frame_rate) * np.random.normal(0,1))
      new_y = self.position_at(-1)[1] + np.sign(direction_to_another_cluster[1]) * np.abs(np.sqrt(2*self.centroid_diffusion_coefficient*self.experiment.frame_rate) * np.random.normal(0,1))

      if not self.cluster_moving_to.distance_to_radio_from(np.array([new_x, new_y])) <= self.cluster_moving_to.distance_to_radio_from(self.position_at(-1)):
        new_x = self.cluster_moving_to.position_at(-1)[0]
        new_y = self.cluster_moving_to.position_at(-1)[1]

    self.positions = np.append(
      self.positions,
      [[
        new_x,
        new_y
      ]], axis=0)

    for particle in self.particles:
      particle.move()
      if particle.cluster is None:
          particles_without_cluster.append(particle)

    for particle in particles_without_cluster: 
      self.particles.remove(particle)

    if self.experiment.save_memory:
      self.positions = np.array([[new_x, new_y]])
    else:
      self.positions = np.append(self.positions, [[new_x, new_y]], axis=0)

    self.lifetime = max(0, self.lifetime-1)

    return particles_without_cluster

  def position_at(self, t):
    return self.positions[t, :]

  def measure_overlap_with(self, another_cluster):
    self_particles_that_are_in_another_cluster = len([particle for particle in self.particles if another_cluster.is_inside(particle=particle)])
    another_particles_that_are_in_self = len([particle for particle in another_cluster.particles if self.is_inside(particle=particle)])

    particle_of_self = len(self.particles)
    particle_of_another_cluster = len(another_cluster.particles)

    return ((self_particles_that_are_in_another_cluster+another_particles_that_are_in_self)/(particle_of_self+particle_of_another_cluster))

  def is_overlapping(self, another_cluster):
    return self.measure_overlap_with(another_cluster) > 0.1

  def move_towards_to(self, another_cluster):
    self.cluster_moving_to = another_cluster
    another_cluster.cluster_moving_to = self

  def can_merge_with(self, another_cluster):
    return another_cluster.cluster_moving_to is not None and another_cluster.cluster_moving_to == self and (self.is_inside_of_cluster(another_cluster) or another_cluster.is_inside_of_cluster(self) or np.array_equal(self.position_at(-1), another_cluster.position_at(-1)))

  def is_inside_of_cluster(self, another_cluster):
    self_particles_that_are_in_another_cluster = len([particle for particle in self.particles if another_cluster.is_inside(particle=particle)])

    particle_of_self = len(self.particles)

    return self_particles_that_are_in_another_cluster/particle_of_self == 1
