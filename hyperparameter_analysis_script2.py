import os

import pandas as pd
from keras import backend as K

from ClusterEdgeRemover import ClusterEdgeRemover
from sklearn.metrics import confusion_matrix, f1_score

print("Number of Hyperparameter Combinations:", len(ClusterEdgeRemover.analysis_hyperparameters()['batch_size']))

results = pd.DataFrame({
    'pos_accuracy': [],
    'neg_accuracy': [],
    'f1-score': [],
    'batch_size': [],
})

DATASET_PATH = 'datasets_shuffled'

#Hyperparameter Analysis
for batch_size in ClusterEdgeRemover.analysis_hyperparameters()['batch_size']:
    print(f"Hyperparameter search with batch size {batch_size}")
    K.clear_session()

    classifier = ClusterEdgeRemover(10,10)

    classifier.hyperparameters['batch_size'] = batch_size

    classifier.fit_with_datasets_from_path(f'./{DATASET_PATH}/train')
    classifier.save_model()
    classifier.test_with_datasets_from_path(f'./{DATASET_PATH}/test', apply_threshold=True, save_result=True, save_predictions=True, check_if_predictions_file_name_exists=True)

    """
    true = []
    pred = []

    for file in os.listdir(f'./{DATASET_PATH}/test/'):
        if file.endswith(f"_predicted_with_batch_size_{classifier.hyperparameters['batch_size']}_{classifier.hyperparameters['radius']}_nofframes_{classifier.hyperparameters['nofframes']}_partition_{classifier.hyperparameters['partition_size']}.csv"):
            dataset = classifier.get_dataset_from_path(f'./{DATASET_PATH}/test/{file}')
            true += dataset['solution'].values.tolist()
            pred += dataset['solution_predicted'].values.tolist()

    if len(true) != 0:
        tn, fp, fn, tp = confusion_matrix(true, pred).ravel()

        results = results.append({
            'pos_accuracy': tn / (tn + fp),
            'neg_accuracy': tp / (tp + fn),
            'batch_size': batch_size,
            'f1-score': f1_score(true, pred)
        }, ignore_index=True)
    """

results.to_csv('results.csv')
