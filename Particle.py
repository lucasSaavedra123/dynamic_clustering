import numpy as np
from stochastic.processes.noise import FractionalGaussianNoise as FGN
import pickle
from functools import lru_cache
from os.path import exists
import itertools

def fgn_autocovariance(hurst, n):
    """Autocovariance function for fGn."""
    ns_2h = np.arange(n + 1) ** (2 * hurst)
    return np.insert((ns_2h[:-2] - 2 * ns_2h[1:-1] + ns_2h[2:]) / 2, 0, 1)

autocovariance = lru_cache(1)(fgn_autocovariance)

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

    #Flagd and 'Blinking Battery'
    self.locked = False
    self.going_out_from_cluster = False
    self.blinking_battery = 0
    self.time_belonging_cluster = experiment.current_time #This only has sense in particle belongs to a cluster
    self.residence_time = residence_time #This only has sense in particle belongs to a cluster

    self.gn_memory_x = None
    self.gn_memory_y = None

    self.anomalous_exponent = np.random.uniform(0,2)

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

        if not self.going_out_from_cluster and (self.cluster.lifetime == 0 or self.experiment.current_time - self.time_belonging_cluster >= self.residence_time):
          self.going_out_from_cluster = True

        if self.going_out_from_cluster:
          retry = True
          while retry:
            displacement_x = self.generate_displacement('x')
            displacement_y = self.generate_displacement('y')

            new_x = displacement_x + self.position_at(-1)[0]
            new_y = displacement_y + self.position_at(-1)[1]

            new_radio = self.cluster.distance_to_radio_from(np.array([new_x, new_y]))
            old_radio = self.cluster.distance_to_radio_from(np.array([self.position_at(-1)[0], self.position_at(-1)[1]]))

            if new_radio < old_radio:
              new_x = -displacement_x + self.position_at(-1)[0]
              new_y = displacement_y + self.position_at(-1)[1]
              new_radio = self.cluster.distance_to_radio_from(np.array([new_x, new_y]))
              if new_radio < old_radio:
                new_x = displacement_x + self.position_at(-1)[0]
                new_y = -displacement_y + self.position_at(-1)[1]
                new_radio = self.cluster.distance_to_radio_from(np.array([new_x, new_y]))
                if new_radio < old_radio:
                  new_x = -displacement_x + self.position_at(-1)[0]
                  new_y = -displacement_y + self.position_at(-1)[1]
                  new_radio = self.cluster.distance_to_radio_from(np.array([new_x, new_y]))
                  if new_radio < old_radio:
                    retry = False
                else:
                  retry = False
              else:
                retry = False
            else:
              retry = False

          if not self.cluster.is_inside(position=np.array([new_x, new_y])):
            self.going_out_from_cluster = False
            self.cluster = None
            self.diffusion_coefficient = np.random.uniform(self.experiment.no_cluster_molecules_diffusion_coefficient_range[0], self.experiment.no_cluster_molecules_diffusion_coefficient_range[1])


        else:
          retry = True
          while retry:

            displacement_x = self.generate_displacement('x')
            displacement_y = self.generate_displacement('y')

            new_x = displacement_x + self.position_at(-1)[0]
            new_y = displacement_y + self.position_at(-1)[1]

            if not self.cluster.is_inside(position=np.array([new_x, new_y])):
              new_x = -displacement_x + self.position_at(-1)[0]
              new_y = displacement_y + self.position_at(-1)[1]
              if not self.cluster.is_inside(position=np.array([new_x, new_y])):
                new_x = displacement_x + self.position_at(-1)[0]
                new_y = -displacement_y + self.position_at(-1)[1]
                if not self.cluster.is_inside(position=np.array([new_x, new_y])):
                  new_x = -displacement_x + self.position_at(-1)[0]
                  new_y = -displacement_y + self.position_at(-1)[1]
                  if self.cluster.is_inside(position=np.array([new_x, new_y])):
                    retry = False
                else:
                  retry = False
              else:
                retry = False
            else:
              retry = False

        if self.can_be_retained and self.cluster is not None and not self.going_out_from_cluster:
          p = self.cluster.probability_to_be_retained(self)
          self.locked = np.random.choice([True, False], 1, p=[p, 1-p])[0]

      else:
        new_x = self.generate_displacement('x') + self.position_at(-1)[0]
        new_y = self.generate_displacement('y') + self.position_at(-1)[1]  

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

  def generate_displacement(self, axis):
    """Generate fractional Gaussian noise using Hosking's method.

    Method of generation is Hosking's method (exact method) from his paper:
    Hosking, J. R. (1984). Modeling persistence in hydrological time series
    using fractional differencing. Water resources research, 20(12),
    1898-1908.

    Hosking's method generates a fractional Gaussian noise (fGn)
    realization. The cumulative sum of this realization gives a fBm.
    """
    # For scaling to interval [0, T]
    increment = 1 / self.experiment.maximum_frame
    hurst = self.anomalous_exponent/2
    scale = increment**hurst

    if axis=='x':
      if self.gn_memory_x is None:
          self.gn_memory_x = np.random.normal(0.0, 1.0, self.experiment.maximum_frame)

      self.gn_memory_x[self.experiment.time] = np.random.normal(0.0, 1.0, 1)
      gn = self.gn_memory_x.copy()
    elif axis=='y':
      if self.gn_memory_y is None:
          self.gn_memory_y = np.random.normal(0.0, 1.0, self.experiment.maximum_frame)

      self.gn_memory_y[self.experiment.time] = np.random.normal(0.0, 1.0, 1)
      gn = self.gn_memory_y.copy()
    # If H = 0.5 then just generate a standard Brownian motion, otherwise
    # proceed with Hosking's method
    if hurst == 0.5:
        fgn = gn
    else:
        # Initializations
        fgn = np.zeros(self.experiment.maximum_frame)
        phi = np.zeros(self.experiment.maximum_frame)
        psi = np.zeros(self.experiment.maximum_frame)

        if not exists(f'./cache/cov_{self.id}'):
          with open(f'./cache/cov_{self.id}', 'wb') as file:
            pickle.dump(autocovariance(hurst, self.experiment.maximum_frame), file)

        with open(f'./cache/cov_{self.id}', 'rb') as f:
            cov = pickle.load(f)

        if not exists(f'./cache/v_memory_{self.id}'):
          v_memory = {}
          phi_memory = {}
          v = 1
          for i in range(1, self.experiment.maximum_frame):
            phi[i - 1] = cov[i]
            for j in range(i - 1):
                psi[j] = phi[j]
                phi[i - 1] -= psi[j] * cov[i - j - 1]
            phi[i - 1] /= v
            for j in range(i - 1):
                phi[j] = psi[j] - phi[i - 1] * psi[i - j - 2]
            
            phi_memory[i] = phi.copy()

            v *= 1 - phi_memory[i][i - 1] * phi_memory[i][i - 1]
            v_memory[i] = v

            with open(f'./cache/v_memory_{self.id}', 'wb') as file:
              pickle.dump(v_memory, file)

            with open(f'./cache/phi_memory_{self.id}', 'wb') as file:
              pickle.dump(phi_memory, file)
       
        with open(f'./cache/phi_memory_{self.id}', 'rb') as f:
            phi_memory = pickle.load(f)

        with open(f'./cache/v_memory_{self.id}', 'rb') as f:
            v_memory = pickle.load(f)

        # First increment from stationary distribution
        fgn[0] = gn[0]
        v = 1
        phi[0] = 0

        if self.experiment.time != 0:
          # Generates fgn realization with n increments of size 1
          for i in range(1, self.experiment.maximum_frame):              
              for j in range(i):
                  fgn[i] += phi_memory[i][j] * fgn[i - j - 1]
              fgn[i] += np.sqrt(v_memory[i]) * gn[i]

              if i == self.experiment.time:
                break

    # Scale to interval [0, T]
    fgn *= scale

    displacement = fgn[self.experiment.time]
    displacement *= np.sqrt(self.experiment.maximum_frame)**(self.anomalous_exponent)
    displacement *= np.sqrt(2*self.diffusion_coefficient*self.experiment.frame_rate)

    return displacement
