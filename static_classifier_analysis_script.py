import os
import time

import pandas as pd

from ClusterEdgeRemover import ClusterEdgeRemover
from LocalizationClassifier import LocalizationClassifier
from utils import predict_on_dataset

DATASET_PATH = './Training Static Datasets'

localization_classifier = LocalizationClassifier(40000,40000, static=True)
localization_classifier.hyperparameters['partition_size'] = 4000

localization_classifier.fit_with_datasets_from_path(DATASET_PATH, save_checkpoints=True)
localization_classifier.load_model()

edge_classifier = ClusterEdgeRemover(40000,40000, static=True)
edge_classifier.hyperparameters['partition_size'] = 4000

#edge_classifier.fit_with_datasets_from_path(DATASET_PATH, save_checkpoints=True)
edge_classifier.load_model()

paths = [
    'Simulated Evaluation Data - 50 PpMS - 10 PpC/50 PpMS - 10 PpC',
    'Simulated Evaluation Data - 100 PpMS - 20PpC/100 PpMS - 20 PpC',
    'Simulated Evaluation Data - 300 PpMS - 100 PpC/300 PpMS - 100 PpC',
]

for path in paths:

    for dataset_file_path in [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.tsv.csv')]:
        print("Predicting for:", dataset_file_path)
        smlm_dataset = pd.read_csv(dataset_file_path)
        st = time.time()
        smlm_dataset = predict_on_dataset(smlm_dataset, localization_classifier, edge_classifier)
        et = time.time()

        with open(dataset_file_path+'_time.txt', "w") as time_file:
            time_file.write(str(et - st))

        smlm_dataset.to_csv(dataset_file_path+".full_prediction.csv", index=False)
