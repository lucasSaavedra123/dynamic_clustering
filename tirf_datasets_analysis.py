import os
import pandas as pd
from utils import predict_on_dataset
import tqdm

from CONSTANTS import *
from LocalizationClassifier import LocalizationClassifier
from ClusterEdgeRemover import ClusterEdgeRemover

localization_classifier = LocalizationClassifier(None,None)
localization_classifier.load_model()

edge_classifier = ClusterEdgeRemover(None,None)
edge_classifier.load_model()

#TEST_DATASETS_PATH = "D:/UCA/03-Clustering Dynamics/SPT-data_TIRF_DATASETS"
TEST_DATASETS_PATH = "./SPT-data_TIRF_DATASETS"

for dataset_file_path in tqdm.tqdm([os.path.join(TEST_DATASETS_PATH, file) for file in os.listdir(TEST_DATASETS_PATH) if file.endswith('_tirf.csv')]):
    smlm_dataset = pd.read_csv(dataset_file_path)

    new_width = smlm_dataset[X_POSITION_COLUMN_NAME].max()
    new_height = smlm_dataset[Y_POSITION_COLUMN_NAME].max()

    localization_classifier.set_dimensions(new_height, new_width)
    edge_classifier.set_dimensions(new_height, new_width)

    smlm_dataset = predict_on_dataset(smlm_dataset, localization_classifier, edge_classifier)
    smlm_dataset.to_csv(dataset_file_path+".full_prediction.csv", index=False)
