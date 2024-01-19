import os
import time

import pandas as pd
import tqdm

from dynamic_clustering.algorithm.ClusterDetector import ClusterDetector
from dynamic_clustering.algorithm.LocalizationClassifier import LocalizationClassifier
from dynamic_clustering.utils import predict_on_dataset, delete_file_if_exist, save_number_in_file


DATASET_PATH = './Training Static Datasets'

localization_classifier = LocalizationClassifier(40000,40000, static=True)
localization_classifier.hyperparameters['partition_size'] = 4000

localization_classifier.fit_with_datasets_from_path(DATASET_PATH, save_checkpoints=True)
localization_classifier.save_model()

#delete_file_if_exist(localization_classifier.train_full_graph_file_name)

edge_classifier = ClusterDetector(40000,40000, static=True)
edge_classifier.hyperparameters['partition_size'] = 4000

edge_classifier.fit_with_datasets_from_path(DATASET_PATH, save_checkpoints=True)
edge_classifier.save_model()

#delete_file_if_exist(edge_classifier.train_full_graph_file_name)

for dataset_file_path in tqdm.tqdm([os.path.join('./Validation Static Datasets', file) for file in os.listdir('./Validation Static Datasets') if file.endswith('.tsv.csv')]):
    if not os.path.isfile(dataset_file_path+".full_prediction.csv"):
        smlm_dataset = pd.read_csv(dataset_file_path)
        st = time.time()
        smlm_dataset = predict_on_dataset(smlm_dataset, localization_classifier, edge_classifier)
        et = time.time()

        #save_number_in_file(dataset_file_path+'_time.txt', et - st)

        #smlm_dataset.to_csv(dataset_file_path+".full_prediction.csv", index=False)
