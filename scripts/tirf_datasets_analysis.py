import os

import pandas as pd
import tqdm

from dynamic_clustering.utils import predict_on_dataset
from dynamic_clustering.algorithm.LocalizationClassifier import LocalizationClassifier
from dynamic_clustering.algorithm.ClusterDetector import ClusterDetector
from dynamic_clustering.CONSTANTS import *


localization_classifier = LocalizationClassifier(None,None)
localization_classifier.load_model()

edge_classifier = ClusterDetector(None,None)
edge_classifier.load_model()

#TEST_DATASETS_PATH = "D:/UCA/03-Clustering Dynamics/TIRF Datasets"
TEST_DATASETS_PATH = "./TIRF Datasets"

for dataset_file_path in tqdm.tqdm([os.path.join(TEST_DATASETS_PATH, file) for file in os.listdir(TEST_DATASETS_PATH) if file.endswith('_tirf.csv')]):
    smlm_dataset = pd.read_csv(dataset_file_path)

    new_width = smlm_dataset[X_POSITION_COLUMN_NAME].max()
    new_height = smlm_dataset[Y_POSITION_COLUMN_NAME].max()

    localization_classifier.set_dimensions(new_height, new_width)
    edge_classifier.set_dimensions(new_height, new_width)

    smlm_dataset = predict_on_dataset(smlm_dataset, localization_classifier, edge_classifier)
    smlm_dataset.to_csv(dataset_file_path+".full_prediction.csv", index=False)
