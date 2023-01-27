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
