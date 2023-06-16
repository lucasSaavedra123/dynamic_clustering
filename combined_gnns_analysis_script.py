import os
import pandas as pd
from utils import predict_on_dataset

from LocalizationClassifier import LocalizationClassifier
from ClusterEdgeRemover import ClusterEdgeRemover

localization_classifier = LocalizationClassifier(10,10)
localization_classifier.load_model()

edge_classifier = ClusterEdgeRemover(10,10)
edge_classifier.load_model()

TEST_DATASETS_PATH = "./datasets_shuffled/test"

for dataset_file_path in [os.path.join(TEST_DATASETS_PATH, file) for file in os.listdir(TEST_DATASETS_PATH) if file.endswith('_smlm_dataset.csv')]:
    print("Predicting for:", dataset_file_path)
    smlm_dataset = pd.read_csv(dataset_file_path)
    smlm_dataset = predict_on_dataset(smlm_dataset, localization_classifier, edge_classifier)
    smlm_dataset.to_csv(dataset_file_path+".full_prediction.csv", index=False)
