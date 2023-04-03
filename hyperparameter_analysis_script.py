import os

import pandas as pd
from keras import backend as K

from LocalizationClassifier import LocalizationClassifier
from sklearn.metrics import confusion_matrix

print("Number of Hyperparameter Combinations:", len(LocalizationClassifier.analysis_hyperparameters()['radius']) * len(LocalizationClassifier.analysis_hyperparameters()['nofframes']) * len(LocalizationClassifier.analysis_hyperparameters()['batch_size']))

results = pd.DataFrame({
    'pos_accuracy': [],
    'neg_accuracy': [],
    'batch_size': [],
    'radius': [],
    'nofframes': []
})

DATASET_PATH = 'datasets_shuffled'

#Hyperparameter Analysis
for radius in LocalizationClassifier.analysis_hyperparameters()['radius']:
    for nofframes in LocalizationClassifier.analysis_hyperparameters()['nofframes']:
        for batch_size in LocalizationClassifier.analysis_hyperparameters()['batch_size']:
            K.clear_session()

            classifier = LocalizationClassifier(10,10)


            classifier.hyperparameters['radius'] = radius
            classifier.hyperparameters['nofframes'] = nofframes
            classifier.hyperparameters['batch_size'] = batch_size

            classifier.fit_with_datasets_from_path(f'./{DATASET_PATH}/train')
            classifier.save_model()
            classifier.test_with_datasets_from_path(f'./{DATASET_PATH}/test', apply_threshold=True, save_result=True, save_predictions=True)

            true = []
            pred = []

            for file in os.listdir(f'./{DATASET_PATH}/test/'):
                if file.endswith(f"_predicted_with_batch_size_{classifier.hyperparameters['batch_size']}_{classifier.hyperparameters['radius']}_nofframes_{classifier.hyperparameters['nofframes']}_partition_{classifier.hyperparameters['partition_size']}.csv"):
                    dataset = classifier.get_dataset_from_path(f'./{DATASET_PATH}/test/{file}')
                    true += dataset['solution'].values.tolist()
                    pred += dataset['solution_predicted'].values.tolist()

            if len(true) != 0:
                tn, fp, fn, tp = confusion_matrix(true, pred).ravel()

                true_negative_rate = tn / (tn + fp)
                true_positive_rate = tp / (tp + fn)

                results = results.append({
                    'pos_accuracy': true_positive_rate,
                    'neg_accuracy': true_negative_rate,
                    'batch_size': batch_size,
                    'radius': radius,
                    'nofframes': nofframes
                }, ignore_index=True)

results.to_csv('results.csv')

"""
classifier = LocalizationClassifier(10,10)
classifier.load_model()
classifier.predict(classifier.get_dataset_from_path('./data/CDx_mab.csv')).to_csv('./data/CDx_mAb_localization_classifier_result_better_with_ghost_threshold_and_increased_partition_size.csv')
"""