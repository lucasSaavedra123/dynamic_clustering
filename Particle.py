import numpy as np
from stochastic.processes.noise import FractionalGaussianNoise as FGN

class Particle():
  def __init__(self, initial_position, diffusion_coefficient, experiment, can_be_retained=False, cluster = None, residence_time = None):
    #Particle state values
    self.positions = np.array([initial_position])
    self.can_be_retained = can_be_retained
    self.cluster = cluster
    self.diffusion_coefficient = diffusion_coefficient
    self.experiment = experiment

    #Flagd and 'Blinking Battery'
    self.locked = False
    self.going_out_from_cluster = False
    self.blinking_battery = 0
    self.time_belonging_cluster = experiment.current_time #This only has sense in particle belongs to a cluster
    self.residence_time = residence_time #This only has sense in particle belongs to a cluster

    self.anomalous_exponent = np.random.uniform(0.1,1.9)

  def position_at(self, t):
    return self.positions[t, :]

  def move(self):
    if self.locked:
      self.positions = np.append(self.positions, [[self.position_at(-1)[0], self.position_at(-1)[1]]], axis=0)

      if self.cluster is not None:
        if not self.cluster.is_inside(self):
          self.cluster = None

          if np.random.choice([False, True], 1, p=[0.50, 0.50])[0]:
            self.locked = False
            self.diffusion_coefficient = np.random.uniform(self.experiment.no_cluster_molecules_diffusion_coefficient_range[0], self.experiment.no_cluster_molecules_diffusion_coefficient_range[1])

    else:
      if self.cluster is not None:
        #El cluster ya se movio!
        original_cluster_direction_movement = self.cluster.position_at(-1) - self.cluster.position_at(-2)
        new_x = self.generate_displacement() + self.position_at(-1)[0] + original_cluster_direction_movement[0]
        new_y = self.generate_displacement() + self.position_at(-1)[1] + original_cluster_direction_movement[1]

        old_radio_from_center = np.linalg.norm(np.array([self.position_at(-1)[0], self.position_at(-1)[1]]) - self.cluster.position_at(-2))
        new_radio_from_center = self.cluster.distance_to_radio_from(np.array([new_x, new_y]))

        if not self.going_out_from_cluster and (self.cluster.lifetime == 0 or self.experiment.current_time - self.time_belonging_cluster >= self.residence_time):
          self.going_out_from_cluster = True

        if self.going_out_from_cluster:
          while new_radio_from_center < old_radio_from_center:
            new_x = self.generate_displacement() + self.position_at(-1)[0] + original_cluster_direction_movement[0]
            new_y = self.generate_displacement() + self.position_at(-1)[1] + original_cluster_direction_movement[1]
            new_radio_from_center = self.cluster.distance_to_radio_from(np.array([new_x, new_y]))

          if not self.cluster.is_inside(position=np.array([new_x, new_y])):
            self.going_out_from_cluster = False
            self.cluster = None
            self.diffusion_coefficient = np.random.uniform(self.experiment.no_cluster_molecules_diffusion_coefficient_range[0], self.experiment.no_cluster_molecules_diffusion_coefficient_range[1])

        else:
          while not self.cluster.is_inside(position=np.array([new_x, new_y])):
            new_x = self.generate_displacement() + self.position_at(-1)[0] + original_cluster_direction_movement[0]
            new_y = self.generate_displacement() + self.position_at(-1)[1] + original_cluster_direction_movement[1]
        
        if self.can_be_retained and self.cluster is not None and not self.going_out_from_cluster:
          p = self.cluster.probability_to_be_retained(self)
          self.locked = np.random.choice([True, False], 1, p=[p, 1-p])[0]

      else:
        new_x = self.generate_displacement() + self.position_at(-1)[0]
        new_y = self.generate_displacement() + self.position_at(-1)[1]  

      if self.experiment.save_memory:
        self.positions = np.array([[new_x, new_y]])
      else:
        self.positions = np.append(self.positions, [[new_x, new_y]], axis=0)

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

  def generate_displacement(self):
    #Code obtained from 
    # https://github.com/AnDiChallenge/andi_datasets/blob/6f9aff34213e89bc7486278269e2746b23369d02/andi_datasets/models_phenom.py#L33
    # Generate displacements
    disp = FGN(hurst = self.anomalous_exponent/2).sample(n = 1)
    # Normalization factor
    disp *= np.sqrt(1)**(self.anomalous_exponent)
    # Add D
    disp *= np.sqrt(2*self.diffusion_coefficient*self.experiment.frame_rate)        
    
    return disp[0]
