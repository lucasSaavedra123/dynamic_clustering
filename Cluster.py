import numpy as np

from Region import *

from Particle import Particle

from hypo import Hypoexponential

class Cluster():
  def is_inside(self, particle):
    return np.linalg.norm(self.position_at(-1) - particle.position_at(-1)) < self.radio

  def distance_to_radio_from(self, position, t=-1):
    return np.linalg.norm(position - self.position_at(t))

  def add_particle(self, particle):
    self.particles.append(particle)
    particle.cluster = self
    factor = np.random.uniform(self.min_factor, self.max_factor)
    particle.diffusion_coefficient = self.centroid_diffusion_coefficient/factor
    particle.can_be_retained=np.random.choice([False, True], 1, p=[0.90, 0.10])
    particle.residence_time = Hypoexponential(self.experiment.residence_time_range).sample(1)[0]
    particle.time_belonging_cluster = self.experiment.current_time

  def __init__(self, radio, initial_position, number_of_initial_particles, centroid_diffusion_coefficient, lifetime, experiment):
    self.radio = radio
    self.outer_region = OuterRegion(self)
    self.inner_region = InnerRegion(self)
    self.center_region = CenterRegion(self)
    self.middle_region = MiddleRegion(self)
    self.number_of_particles_leaving_cluster = 0
    self.positions = np.array([initial_position])
    self.particles = []
    self.experiment = experiment
    self.number_of_particles_going_out = 0

    self.lifetime = lifetime

    self.cluster_change_direction = np.random.choice([-1, 1], 1, p=[0.5, 0.5])[0]

    self.centroid_diffusion_coefficient = centroid_diffusion_coefficient

    self.slower_than_particles = np.random.choice([False, True], 1, p=[0.5, 0.5])[0]

    if self.slower_than_particles:
      self.min_factor = 0.5
      self.max_factor = 1
    else:
      self.min_factor = 1
      self.max_factor = 2

    for i in range(number_of_initial_particles):
      angle = np.random.uniform(0, 2 * np.pi)
      radio = np.random.uniform(0, self.radio)

      self.particles.append(
          Particle(
            [initial_position[0] + np.cos(angle) * radio, initial_position[1] + np.sin(angle) * radio],
            self.centroid_diffusion_coefficient / np.random.uniform(self.min_factor, self.max_factor),
            experiment,
            can_be_retained=np.random.choice([False, True], 1, p=[0.95, 0.05]),
            cluster=self,
            residence_time=Hypoexponential(self.experiment.residence_time_range).sample(1)[0]
          )
        )

  def probability_to_be_retained(self, radio):
    if self.outer_region.inside_region(radio):
      return self.outer_region.probability_to_be_retained
    elif self.inner_region.inside_region(radio):
      return self.inner_region.probability_to_be_retained
    elif self.center_region.inside_region(radio):
      return self.center_region.probability_to_be_retained
    elif self.middle_region.inside_region(radio):
      return self.middle_region.probability_to_be_retained
    else:
      return 0

  def move(self):
    particles_without_cluster = []

    new_x = self.position_at(-1)[0] + np.sqrt(2*self.centroid_diffusion_coefficient*self.experiment.frame_rate) * np.random.normal(0,1)
    new_y = self.position_at(-1)[1] + np.sqrt(2*self.centroid_diffusion_coefficient*self.experiment.frame_rate) * np.random.normal(0,1)

    self.positions = np.append(
        self.positions,
        [[
          new_x,
          new_y
        ]], axis=0)

    self.radio = max(min(self.experiment.radio_range), min(max(self.experiment.radio_range), self.radio + self.cluster_change_direction * 0.001))

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
