import numpy as np

class Region():
  def __init__(self, cluster):
    self.cluster = cluster
    self.min_radio = self.minimum_assign_radio()
    self.max_radio = self.maximum_assign_radio()
  
  def inside_region(self, radio):
    return self.min_radio < radio < self.max_radio

class OuterRegion(Region):
  def maximum_assign_radio(self):
    return self.cluster.radio + self.cluster.radio * 0.15

  def minimum_assign_radio(self):
    return self.cluster.radio

  @property
  def probability_to_be_retained(self):
    return 0

class InnerRegion(Region):
  def maximum_assign_radio(self):
    return self.cluster.radio

  def minimum_assign_radio(self):
    return self.cluster.radio - self.cluster.radio * 0.15

  @property
  def probability_to_be_retained(self):
    return 0.05

class MiddleRegion(Region):
  def maximum_assign_radio(self):
    return self.cluster.inner_region.min_radio

  def minimum_assign_radio(self):
    return self.cluster.center_region.max_radio

  @property
  def probability_to_be_retained(self):
    return 0.1

class CenterRegion(Region):
  def maximum_assign_radio(self):
    return np.random.uniform(self.cluster.radio * 0.05, self.cluster.radio * 0.2)

  def minimum_assign_radio(self):
    return 0

  @property
  def probability_to_be_retained(self):
    return 0.5

class RetentionProbabilityWithDiscreteFunction():
  def __call__(self, particle):
    """
    if self.cluster.outer_region.inside_region(radio):
      return self.cluster.outer_region.probability_to_be_retained
    elif self.cluster.inner_region.inside_region(radio):
      return self.cluster.inner_region.probability_to_be_retained
    elif self.cluster.middle_region.inside_region(radio):
      return self.cluster.middle_region.probability_to_be_retained
    elif self.cluster.center_region.inside_region(radio):
      return self.cluster.center_region.probability_to_be_retained
    else:
      return 0
    """

    return 0

class RetentionProbabilityWithCuadraticFunction():

  def __call__(self, particle):
    max_probability = particle.experiment.max_retention_probability

    if particle is not None:
      x = particle.position_at(-1)[0]
      y = particle.position_at(-1)[1]

    # The ellipse
    g_ell_center = particle.cluster.position_at(-1)
    g_ell_width = particle.cluster.width
    g_ell_height = particle.cluster.height
    angle = particle.cluster.angle

    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    xc = x - g_ell_center[0]
    yc = y - g_ell_center[1]

    xct = xc * cos_angle - yc * sin_angle
    yct = xc * sin_angle + yc * cos_angle 
  
    print("From Cuadratic...")
    return max(0, - ((xct/np.sqrt(1/max_probability))**2/(g_ell_width/2.)**2) - ((yct/np.sqrt(1/max_probability))**2/(g_ell_height/2.)**2) + max_probability)

class RetentionProbabilityWithLinearFunction():
  def __call__(self, particle):

    max_probability = particle.experiment.max_retention_probability

    if particle is not None:
      x = particle.position_at(-1)[0]
      y = particle.position_at(-1)[1]

    # The ellipse
    g_ell_center = particle.cluster.position_at(-1)
    g_ell_width = particle.cluster.width
    g_ell_height =particle.cluster.height
    angle = particle.cluster.angle

    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    xc = x - g_ell_center[0]
    yc = y - g_ell_center[1]

    xct = xc * cos_angle - yc * sin_angle
    yct = xc * sin_angle + yc * cos_angle 

    print("From Linear...")
    return max(0 , np.sqrt((xct**2/(g_ell_width/2.)**2) + (yct**2/(g_ell_height/2.)**2)  * ((g_ell_height/2)**2)) + max_probability)
