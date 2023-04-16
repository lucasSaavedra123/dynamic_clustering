import os

import pandas as pd
from keras import backend as K

from ClusterEdgeRemover import ClusterEdgeRemover
from sklearn.metrics import confusion_matrix, f1_score

print("Number of Hyperparameter Combinations:", len(ClusterEdgeRemover.analysis_hyperparameters()['batch_size']))

"""
results = pd.DataFrame({
    'pos_accuracy': [],
    'neg_accuracy': [],
    'f1-score': [],
    'batch_size': [],
})
"""

DATASET_PATH = 'datasets_shuffled'

#Hyperparameter Analysis
for batch_size in ClusterEdgeRemover.analysis_hyperparameters()['batch_size']:
    print(f"Hyperparameter search with batch size {batch_size}")
    K.clear_session()

    classifier = ClusterEdgeRemover(10,10)

    classifier.hyperparameters['batch_size'] = batch_size

    classifier.fit_with_datasets_from_path(f'./{DATASET_PATH}/train')
    classifier.save_model()
    classifier.test_with_datasets_from_path(f'./{DATASET_PATH}/test', save_result=True, save_predictions=True, apply_threshold=True, check_if_predictions_file_name_exists=True)

#results.to_csv('results.csv')
