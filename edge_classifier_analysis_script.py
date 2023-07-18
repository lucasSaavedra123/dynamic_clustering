from keras import backend as K
from ClusterEdgeRemover import ClusterEdgeRemover


DATASET_PATH = 'datasets_shuffled'

for partition_size in ClusterEdgeRemover.analysis_hyperparameters()['partition_size']:
    print(f"Hyperparameter search with partition size {partition_size}")
    K.clear_session()

    classifier = ClusterEdgeRemover(10,10)
    classifier.hyperparameters['partition_size'] = partition_size

    classifier.fit_with_datasets_from_path(f'./{DATASET_PATH}/train', save_checkpoints=True)
    classifier.save_model()

for partition_size in ClusterEdgeRemover.analysis_hyperparameters()['partition_size']:
    print(f"Testing with partition size {partition_size}")

    classifier = ClusterEdgeRemover(10,10)
    classifier.hyperparameters['partition_size'] = partition_size
    classifier.load_model()
    """
    if partition_size == 4000:
        classifier.test_with_datasets_from_path(f'./{DATASET_PATH}/test', save_result=False, detect_clusters=True, save_predictions=True, apply_threshold=True, check_if_predictions_file_name_exists=False, ignore_non_clustered_localizations=True)
    """
    classifier.test_with_datasets_from_path(f'./{DATASET_PATH}/test', save_result=True, detect_clusters=False, save_predictions=False, apply_threshold=True, check_if_predictions_file_name_exists=True, ignore_non_clustered_localizations=True)
