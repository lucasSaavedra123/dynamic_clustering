import os
import pandas as pd

from LocalizationClassifier import LocalizationClassifier
from ClusterEdgeRemover import ClusterEdgeRemover

localization_classifier = LocalizationClassifier(10,10)
localization_classifier.hyperparameters['partition_size'] = 3000
localization_classifier.load_model()

edge_classifier = ClusterEdgeRemover(10,10)
edge_classifier.hyperparameters['partition_size'] = 4000
edge_classifier.load_model()

TEST_DATASETS_PATH = './datasets_shuffled/test'

for dataset_file_path in [os.path.join(TEST_DATASETS_PATH, file) for file in os.listdir(TEST_DATASETS_PATH) if file.endswith('_smlm_dataset.csv')]:
    print("Predicting for:", dataset_file_path)

    smlm_dataset = pd.read_csv(dataset_file_path)

    magik_dataset = localization_classifier.transform_smlm_dataset_to_magik_dataframe(smlm_dataset)
    magik_dataset = localization_classifier.predict(magik_dataset)
    smlm_dataset = localization_classifier.transform_magik_dataframe_to_smlm_dataset(magik_dataset)

    magik_dataset = edge_classifier.transform_smlm_dataset_to_magik_dataframe(smlm_dataset)
    magik_dataset = edge_classifier.predict(magik_dataset, detect_clusters=True, apply_threshold=True, original_dataset_path=dataset_file_path)
    smlm_dataset = edge_classifier.transform_magik_dataframe_to_smlm_dataset(magik_dataset)

    smlm_dataset.to_csv(dataset_file_path+".full_prediction.csv", index=False)
