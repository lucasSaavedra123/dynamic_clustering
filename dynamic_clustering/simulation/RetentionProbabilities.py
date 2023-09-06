import numpy as np

class Region():
  def __init__(self, cluster):
    self.cluster = cluster
    self.height = self.assign_height()
    self.width = self.assign_width()
    self.angle = cluster.angle

  def inside_region(self, particle):
    x = particle.position_at(-1)[0]
    y = particle.position_at(-1)[1]

    g_ell_center = particle.cluster.position_at(-1)
    g_ell_width = self.width
    g_ell_height = self.height
    angle = self.angle

    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    xc = x - g_ell_center[0]
    yc = y - g_ell_center[1]

    xct = xc * cos_angle - yc * sin_angle
    yct = xc * sin_angle + yc * cos_angle 

    return (xct**2/(g_ell_width/2.)**2) + (yct**2/(g_ell_height/2.)**2) <= 1


class InnerRegion(Region):
  def assign_height(self):
    return self.cluster.height - self.cluster.height * 0.10

  def assign_width(self):
    return self.cluster.width - self.cluster.width * 0.10

class MiddleRegion(Region):
  def assign_height(self):
    return self.cluster.height - self.cluster.height * 0.80

  def assign_width(self):
    return self.cluster.width - self.cluster.width * 0.80

class RetentionProbabilityWithDiscreteFunction():
  def __call__(self, particle):

    self.build_regions(particle)

    if particle.cluster.inner_region.inside_region(particle) and particle.cluster.middle_region.inside_region(particle):
      return 0.5
    elif particle.cluster.inner_region.inside_region(particle):
      return 0.1
    else:
      return 0.01

  def build_regions(self, particle):
    particle.cluster.inner_region = InnerRegion(particle.cluster)
    particle.cluster.middle_region = MiddleRegion(particle.cluster)

class RetentionProbabilityWithCuadraticFunction():

  def __call__(self, particle):
    max_probability = particle.experiment.max_retention_probability
    min_probability = particle.experiment.min_retention_probability

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
  
    return max(0, - ((xct/np.sqrt(4/(max_probability-min_probability)))**2/(g_ell_width/2.)**2) - ((yct/np.sqrt(4/(max_probability-min_probability)))**2/(g_ell_height/2.)**2) + max_probability)

class RetentionProbabilityWithLinearFunction():
  def __call__(self, particle):

    max_probability = particle.experiment.max_retention_probability
    min_probability = particle.experiment.min_retention_probability

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

    return max(0 , - np.sqrt((((xct/(2/(max_probability-min_probability)))**2/(g_ell_width/2.)**2) + ((yct/(2/(max_probability-min_probability)))**2/(g_ell_height/2.)**2))) + max_probability)
