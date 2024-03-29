from keras import backend as K

from dynamic_clustering.algorithm.LocalizationClassifier import LocalizationClassifier


DATASET_PATH = 'datasets_shuffled'

for partition_size in LocalizationClassifier.analysis_hyperparameters()['partition_size']:
    print(f"Hyperparameter search with partition size {partition_size}")
    K.clear_session()

    classifier = LocalizationClassifier(10,10)
    classifier.hyperparameters['partition_size'] = partition_size

    classifier.fit_with_datasets_from_path(f'./{DATASET_PATH}/train', save_checkpoints=True)
    classifier.save_model()

for partition_size in LocalizationClassifier.analysis_hyperparameters()['partition_size']:
    print(f"Testing with partition size {partition_size}")

    classifier = LocalizationClassifier(10,10)
    classifier.hyperparameters['partition_size'] = partition_size

    classifier.load_model()
    classifier.test_with_datasets_from_path(f'./{DATASET_PATH}/test', apply_threshold=True, save_result=True, save_predictions=True, check_if_predictions_file_name_exists=True)
