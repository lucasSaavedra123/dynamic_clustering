import numpy as np

from Particle import Particle

from hypo import Hypoexponential

class Cluster():
  """
  def is_inside(self, particle):
    return np.linalg.norm(self.position_at(-1) - particle.position_at(-1)) < self.radio
  """
  
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
    return np.linalg.norm(position - self.position_at(t))

  def add_particle(self, particle):
    self.particles.append(particle)
    particle.cluster = self
    factor = np.random.uniform(self.min_factor, self.max_factor)
    particle.diffusion_coefficient = self.centroid_diffusion_coefficient/factor
    particle.can_be_retained=np.random.choice([False, True], 1, p=[0.90, 0.10])
    particle.residence_time = Hypoexponential(self.experiment.residence_time_range).sample(1)[0]
    particle.time_belonging_cluster = self.experiment.current_time

  def __init__(self, radio, initial_position, number_of_initial_particles, centroid_diffusion_coefficient, retention_probability_function, lifetime, eccentricity_maximum, experiment):
    self.radio = radio
    self.number_of_particles_leaving_cluster = 0
    self.positions = np.array([initial_position])
    self.particles = []
    self.experiment = experiment
    self.number_of_particles_going_out = 0
    self.eccentricity_maximum = eccentricity_maximum
    self.retention_probability_function = retention_probability_function() # We create an instance of it

    bad_initial_shape = True

    self.angle = np.random.uniform(0, 2*np.pi)

    while bad_initial_shape:
      self.width = np.random.uniform(self.experiment.radio_range[0], self.experiment.radio_range[1]) * 2
      self.height = np.random.uniform(self.experiment.radio_range[0], self.experiment.radio_range[1]) * 2

      if self.width < radio * 2 or self.height < radio * 2 or self.eccentricity > self.eccentricity_maximum:
        bad_initial_shape = True
      else:
        bad_initial_shape = False

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

    for _ in range(number_of_initial_particles):
      not_inside = True

      while not_inside:
        angle = np.random.uniform(0, 2 * np.pi)
        radio = np.random.uniform(0, max(self.height/2, self.width/2))
        new_position = [initial_position[0] + np.cos(angle) * radio, initial_position[1] + np.sin(angle) * radio]

        if self.is_inside(position=new_position):
          not_inside = False

      self.particles.append(
          Particle(
            new_position,
            self.centroid_diffusion_coefficient / np.random.uniform(self.min_factor, self.max_factor),
            experiment,
            can_be_retained=np.random.choice([False, True], 1, p=[0.95, 0.05]),
            cluster=self,
            residence_time=Hypoexponential(self.experiment.residence_time_range).sample(1)[0]
          )
        )

  def probability_to_be_retained(self, particle):
    assert particle in self.particles
    return self.retention_probability_function(particle)

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

    self.change_cluster_shape()

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

  def change_cluster_shape(self):
    valid_new_shape = False

    while not valid_new_shape:
      new_width = self.width + np.random.normal(0, 0.0001)
      new_height = self.height + np.random.normal(0, 0.0001)
      new_angle = self.angle + np.random.normal(0, 0.01)

      if new_width < self.radio * 2 or new_height < self.radio * 2 or self.eccentricity > self.eccentricity_maximum:
        valid_new_shape = False
      else:
        valid_new_shape = True

    self.width = max(new_width, max(self.experiment.radio_range))
    self.height = max(new_height, max(self.experiment.radio_range))
    self.angle = new_angle
