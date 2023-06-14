from math import sqrt
from sklearn.neighbors import NearestNeighbors

from CONSTANTS import *

def custom_norm(vector_one, vector_two):
  a = pow(vector_one[0] - vector_two[0], 2)
  b = pow(vector_one[1] - vector_two[1], 2)
  #assert np.linalg.norm(vector_one-vector_two) == sqrt(a+b)
  return sqrt(a+b)

def predict_on_dataset(smlm_dataset, localization_classifier, edge_classifier):
    TEMPORAL_FILE_NAME = 'for_delete.for_delete'

    magik_dataset = localization_classifier.transform_smlm_dataset_to_magik_dataframe(smlm_dataset)
    magik_dataset = localization_classifier.predict(magik_dataset, apply_threshold=True)
    smlm_dataset = localization_classifier.transform_magik_dataframe_to_smlm_dataset(magik_dataset)

    smlm_dataset.to_csv(TEMPORAL_FILE_NAME)

    magik_dataset = edge_classifier.transform_smlm_dataset_to_magik_dataframe(smlm_dataset)
    magik_dataset = edge_classifier.predict(magik_dataset, detect_clusters=True, apply_threshold=True, original_dataset_path=TEMPORAL_FILE_NAME)
    magik_dataset[CLUSTERIZED_COLUMN_NAME + "_predicted"] =  (magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED] != 0).astype(int) #Revalidation
    smlm_dataset = edge_classifier.transform_magik_dataframe_to_smlm_dataset(magik_dataset)

    """
    columns_to_pick = [MAGIK_X_POSITION_COLUMN_NAME, MAGIK_Y_POSITION_COLUMN_NAME, TIME_COLUMN_NAME]
    retry = True

    while retry:
        nbrs = NearestNeighbors(n_neighbors=2, n_jobs=-1).fit(magik_dataset[columns_to_pick].values)

        localizations_classifier_as_positive = magik_dataset[magik_dataset[CLUSTERIZED_COLUMN_NAME+'_predicted'] == 1]

        if len(localizations_classifier_as_positive) != 0:
            _, indices = nbrs.kneighbors(localizations_classifier_as_positive[columns_to_pick].values)

            left_index = magik_dataset.iloc[indices[:,0]].index
            right_index = magik_dataset.iloc[indices[:,1]].index

            magik_dataset.loc[left_index, CLUSTERIZED_COLUMN_NAME+'_predicted'] = magik_dataset.loc[right_index, CLUSTERIZED_COLUMN_NAME+'_predicted'].values

            new_localizations_classifier_as_positive = magik_dataset[magik_dataset[CLUSTERIZED_COLUMN_NAME+'_predicted'] == 1]
            retry = not new_localizations_classifier_as_positive.equals(localizations_classifier_as_positive)
        else:
            retry = False

    magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED] =  magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED] *  magik_dataset[CLUSTERIZED_COLUMN_NAME+'_predicted']

    smlm_dataset = edge_classifier.transform_magik_dataframe_to_smlm_dataset(magik_dataset)

    smlm_dataset.to_csv(TEMPORAL_FILE_NAME)

    magik_dataset = edge_classifier.transform_smlm_dataset_to_magik_dataframe(smlm_dataset)
    magik_dataset = edge_classifier.predict(magik_dataset, detect_clusters=True, apply_threshold=True, original_dataset_path=TEMPORAL_FILE_NAME)

    smlm_dataset = edge_classifier.transform_magik_dataframe_to_smlm_dataset(magik_dataset)

    magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED] =  magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED] *  magik_dataset[CLUSTERIZED_COLUMN_NAME+'_predicted']
    """

    return smlm_dataset
