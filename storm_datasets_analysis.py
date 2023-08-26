import os
import pandas as pd
from utils import predict_on_dataset
import tqdm

from LocalizationClassifier import LocalizationClassifier
from ClusterDetector import ClusterDetector

localization_classifier = LocalizationClassifier(10,10)
localization_classifier.load_model()

edge_classifier = ClusterDetector(10,10)
edge_classifier.load_model()

#TEST_DATASETS_PATH = "D:/UCA/03-Clustering Dynamics/STORM Datasets"
TEST_DATASETS_PATH = "./STORM Datasets"

for dataset_file_path in tqdm.tqdm([os.path.join(TEST_DATASETS_PATH, file) for file in os.listdir(TEST_DATASETS_PATH) if file.endswith('_smlm_dataset.csv')]):
    smlm_dataset = pd.read_csv(dataset_file_path)
    smlm_dataset = predict_on_dataset(smlm_dataset, localization_classifier, edge_classifier)
    smlm_dataset.to_csv(dataset_file_path+".full_prediction.csv", index=False)
