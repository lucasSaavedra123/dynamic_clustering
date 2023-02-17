import numpy as np
from functools import lru_cache
import itertools
from TrajectoryDisplacementGenerator import TrajectoryDisplacementGenerator

class Particle():
  id_obj = itertools.count(1)
  def __init__(self, initial_position, diffusion_coefficient, experiment, can_be_retained=False, cluster = None, residence_time = None):
    #Particle state values
    self.positions = np.array([initial_position])
    self.id = next(Particle.id_obj)
    self.can_be_retained = can_be_retained
    self.cluster = cluster
    self.diffusion_coefficient = diffusion_coefficient
    self.experiment = experiment

    self.previous_cluster = None

    #Flagd and 'Blinking Battery'
    self.locked = False
    self.going_out_from_cluster = False
    self.blinking_battery = 0
    self.time_belonging_cluster = experiment.current_time #This only has sense in particle belongs to a cluster
    self.residence_time = residence_time #This only has sense in particle belongs to a cluster

    self.was_inside = None #Neccesary when clusters are merged

    self.direction = 1

    self.anomalous_exponent = np.random.uniform(experiment.anomalous_exponent_range[0], experiment.anomalous_exponent_range[1])

    self.displacement_generator_x = TrajectoryDisplacementGenerator(self.anomalous_exponent, self.experiment.maximum_frame)
    self.displacement_generator_x_as_iterator = iter(self.displacement_generator_x)

    self.displacement_generator_y = TrajectoryDisplacementGenerator(self.anomalous_exponent, self.experiment.maximum_frame)
    self.displacement_generator_y_as_iterator = iter(self.displacement_generator_y)

  def in_fov(self):
    position = self.position_at(-1)
    
    is_horizontally = 0 <= position[0] <= self.experiment.width
    is_vertically = 0 <= position[1] <= self.experiment.height

    return is_horizontally and is_vertically

  def position_at(self, t):
    return self.positions[t, :]

  def _move_closer(self):
    old_radio = self.cluster.distance_to_radio_from(self.position_at(-1))
    retry = True
    while retry:
      displacement_x = self.generate_displacement('x')
      displacement_y = self.generate_displacement('y')

      self.new_x = self.direction * displacement_x + self.position_at(-1)[0]
      self.new_y = self.direction * displacement_y + self.position_at(-1)[1]
      new_radio = self.cluster.distance_to_radio_from(np.array([self.new_x, self.new_y]))

      if new_radio > old_radio:
        self.direction = -self.direction
        self.new_x = self.direction * displacement_x + self.position_at(-1)[0]
        self.new_y = self.direction * displacement_y + self.position_at(-1)[1]
        new_radio = self.cluster.distance_to_radio_from(np.array([self.new_x, self.new_y]))
        if not (new_radio > old_radio):
          retry = False
      else:
        retry = False

  def _move_further(self):
    old_radio = self.cluster.distance_to_radio_from(self.position_at(-1))
    retry = True
    while retry:
      displacement_x = self.generate_displacement('x')
      displacement_y = self.generate_displacement('y')

      self.new_x = self.direction * displacement_x + self.position_at(-1)[0]
      self.new_y = self.direction * displacement_y + self.position_at(-1)[1]

      new_radio = self.cluster.distance_to_radio_from(np.array([self.new_x, self.new_y]))

      if new_radio < old_radio:
        self.direction = -self.direction
        self.new_x = self.direction * displacement_x + self.position_at(-1)[0]
        self.new_y = self.direction * displacement_y + self.position_at(-1)[1]
        new_radio = self.cluster.distance_to_radio_from(np.array([self.new_x, self.new_y]))
        if not (new_radio < old_radio):
          retry = False
      else:
        retry = False

  def _move_inside_cluster(self):
    retry = True
    while retry:
      displacement_x = self.generate_displacement('x')
      displacement_y = self.generate_displacement('y')

      self.new_x = self.direction * displacement_x + self.position_at(-1)[0]
      self.new_y = self.direction * displacement_y + self.position_at(-1)[1]

      if not self.cluster.is_inside(position=np.array([self.new_x, self.new_y])):
        self.direction = -self.direction
        self.new_x = self.direction * displacement_x + self.position_at(-1)[0]
        self.new_y = self.direction * displacement_y + self.position_at(-1)[1]
        if self.cluster.is_inside(position=np.array([self.new_x, self.new_y])):
          retry = False
      else:
        retry = False
  
  def _move_as_locked(self):
    self.new_x = self.generate_displacement('x') * 0 + self.position_at(-1)[0]
    self.new_y = self.generate_displacement('y') * 0 + self.position_at(-1)[1]

    if self.cluster is not None:
      if not self.cluster.is_inside(self):
        self.previous_cluster = self.cluster
        self.cluster = None

        if np.random.choice([False, True], 1, p=[0.50, 0.50])[0]:
          self.locked = False
          self.diffusion_coefficient = np.random.uniform(self.experiment.no_cluster_molecules_diffusion_coefficient_range[0], self.experiment.no_cluster_molecules_diffusion_coefficient_range[1])

  def _move_as_free(self):
    self.new_x = self.generate_displacement('x') + self.position_at(-1)[0]
    self.new_y = self.generate_displacement('y') + self.position_at(-1)[1]

  def _move_as_clustered(self):
    #cluster_displacement = self.cluster.position_at(-1) - self.cluster.position_at(-2)
    if not self.going_out_from_cluster and (self.cluster.lifetime == 0 or self.experiment.current_time - self.time_belonging_cluster >= self.residence_time):
      self.going_out_from_cluster = True

    if self.going_out_from_cluster:
      self._move_further()

      if not self.cluster.is_inside(position=np.array([self.new_x, self.new_y])):
        self.going_out_from_cluster = False
        self.previous_cluster = self.cluster
        self.cluster = None
        self.diffusion_coefficient = np.random.uniform(self.experiment.no_cluster_molecules_diffusion_coefficient_range[0], self.experiment.no_cluster_molecules_diffusion_coefficient_range[1])
    else:
      if self.cluster.is_inside(position=self.position_at(-1)):
        self._move_inside_cluster()
      else:
        self._move_closer()

    if self.cluster is not None and self.can_be_retained and not self.going_out_from_cluster:
      p = self.cluster.probability_to_be_retained(self)
      self.locked = np.random.choice([True, False], 1, p=[p, 1-p])[0]

  def move(self):
    if self.locked:
      self._move_as_locked()
    else:
      if self.cluster is not None:
        self._move_as_clustered()
      else:
        self._move_as_free()  

    if self.experiment.save_memory:
      self.positions = np.array([[self.new_x, self.new_y]])
    else:
      self.positions = np.append(self.positions, [[self.new_x, self.new_y]], axis=0)

    self.displacement_generator_x.next_step()
    self.displacement_generator_y.next_step()

    self.blinking_battery = max(self.blinking_battery-1,0)

  @property
  def color(self):
    if self.locked:
      return 'blue'
    elif self.going_out_from_cluster:
      return 'violet'
    elif self.cluster is None:
      return 'black'
    else:
      return 'red'

  def generate_displacement(self, axis):
    if axis=='x':
      generated_displacement = next(self.displacement_generator_x_as_iterator)
    elif axis=='y':
      generated_displacement = next(self.displacement_generator_y_as_iterator)

    generated_displacement *= np.sqrt(self.experiment.maximum_frame)**(self.anomalous_exponent)
    generated_displacement *= np.sqrt(2*self.diffusion_coefficient*self.experiment.frame_rate)

    return generated_displacement

  def came_from_existent_cluster(self):
    if self.previous_cluster is not None:
      if self.previous_cluster.exist:
        return True
      else:
        self.previous_cluster = None
        return False
    else:
      return False
