from math import sqrt


def custom_norm(vector_one, vector_two):
  a = pow(vector_one[0] - vector_two[0], 2)
  b = pow(vector_one[1] - vector_two[1], 2)
  #assert np.linalg.norm(vector_one-vector_two) == sqrt(a+b)
  return sqrt(a+b)

def custom_mean(vector):

  number_of_elements = 0
  sum = 0

  for i in vector:
    sum += i
    number_of_elements += 1

  return sum/number_of_elements
