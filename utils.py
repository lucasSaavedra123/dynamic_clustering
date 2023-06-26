from math import sqrt
import os

from scipy.spatial import Delaunay

from CONSTANTS import *


def custom_norm(vector_one, vector_two):
  a = pow(vector_one[0] - vector_two[0], 2)
  b = pow(vector_one[1] - vector_two[1], 2)
  #assert np.linalg.norm(vector_one-vector_two) == sqrt(a+b)
  return sqrt(a+b)

def delaunay_from_dataframe(dataframe, columns_to_pick):
  list_of_edges = []

  simplices = Delaunay(dataframe[columns_to_pick].values).simplices

  def less_first(a, b):
      return [a,b] if a < b else [b,a]

  for triangle in simplices:
      for e1, e2 in [[0,1],[1,2],[2,0]]: # for all edges of triangle
          list_of_edges.append(less_first(triangle[e1],triangle[e2])) # always lesser index first

  return list_of_edges

def predict_on_dataset(smlm_dataset, localization_classifier, edge_classifier):
    TEMPORAL_FILE_NAME = 'for_delete.for_delete'

    magik_dataset = localization_classifier.transform_smlm_dataset_to_magik_dataframe(smlm_dataset)
    magik_dataset = localization_classifier.predict(magik_dataset, apply_threshold=True)
    smlm_dataset = localization_classifier.transform_magik_dataframe_to_smlm_dataset(magik_dataset)

    smlm_dataset.to_csv(TEMPORAL_FILE_NAME)

    magik_dataset = edge_classifier.transform_smlm_dataset_to_magik_dataframe(smlm_dataset)
    magik_dataset = edge_classifier.predict(magik_dataset, detect_clusters=True, apply_threshold=True, original_dataset_path=TEMPORAL_FILE_NAME)
    smlm_dataset = edge_classifier.transform_magik_dataframe_to_smlm_dataset(magik_dataset)

    os.remove(TEMPORAL_FILE_NAME)

    return smlm_dataset
