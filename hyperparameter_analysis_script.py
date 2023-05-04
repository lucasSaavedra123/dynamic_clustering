import os

import pandas as pd
from keras import backend as K
from sklearn.metrics import confusion_matrix, f1_score

from LocalizationClassifier import LocalizationClassifier

results = pd.DataFrame({
    'pos_accuracy': [],
    'neg_accuracy': [],
    'f1-score': [],
    'batch_size': [],
})

DATASET_PATH = 'datasets_shuffled'

#Hyperparameter Analysis
for batch_size in LocalizationClassifier.analysis_hyperparameters()['batch_size']:
    print(f"Hyperparameter search with batch size {batch_size}")
    for partition_size in [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]:
        K.clear_session()

        classifier = LocalizationClassifier(10,10)

        classifier.hyperparameters['batch_size'] = batch_size
        classifier.hyperparameters['partition_size'] = partition_size

        classifier.fit_with_datasets_from_path(f'./{DATASET_PATH}/train')
        classifier.save_model()
        classifier.test_with_datasets_from_path(f'./{DATASET_PATH}/test', apply_threshold=True, save_result=True, save_predictions=True, check_if_predictions_file_name_exists=True)

        true = []
        pred = []

        for file in os.listdir(f'./{DATASET_PATH}/test/'):
            if file.endswith(f"_predicted_with_batch_size_{classifier.hyperparameters['batch_size']}_{classifier.hyperparameters['radius']}_nofframes_{classifier.hyperparameters['nofframes']}_partition_{classifier.hyperparameters['partition_size']}.csv"):
                dataset = classifier.get_dataset_from_path(f'./{DATASET_PATH}/test/{file}')

                if len(dataset[dataset['solution'] == 1]) != 0:
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

    results.to_csv('localization_classifier_results.csv')
