import os
import time
from utils import *

import pandas as pd
import tqdm

from LocalizationClassifier import LocalizationClassifier
from ClusterEdgeRemover import ClusterEdgeRemover


localization_classifier = LocalizationClassifier(10,10)
localization_classifier.load_model()

edge_classifier = ClusterEdgeRemover(10,10)
edge_classifier.load_model()

TEST_DATASETS_PATH = "./datasets_shuffled/test"

for partition_size in [500,1000,1500,2000,2500,3000,3500,4000]:
    print("Partition Size:", partition_size)

    localization_classifier.hyperparameters['partition_size'] = partition_size
    edge_classifier.hyperparameters['partition_size'] = partition_size

    gnn1_times = []
    gnn2_times = []

    for dataset_file_path in tqdm.tqdm([os.path.join(TEST_DATASETS_PATH, file) for file in os.listdir(TEST_DATASETS_PATH) if file.endswith('_smlm_dataset.csv')]):
        magik_dataset = localization_classifier.transform_smlm_dataset_to_magik_dataframe(pd.read_csv(dataset_file_path))
        st = time.time()
        localization_classifier.predict(magik_dataset)
        et = time.time()
        gnn1_times.append(et-st)

        magik_dataset = edge_classifier.transform_smlm_dataset_to_magik_dataframe(pd.read_csv(dataset_file_path))
        st = time.time()
        edge_classifier.predict(magik_dataset, detect_clusters=False)
        et = time.time()
        gnn2_times.append(et-st)

    save_numbers_in_file(f'gnn1_partition_size_{partition_size}_times.txt', gnn1_times)
    save_numbers_in_file(f'gnn2_partition_size_{partition_size}_times.txt', gnn2_times)
