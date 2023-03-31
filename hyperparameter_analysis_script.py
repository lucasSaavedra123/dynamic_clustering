from LocalizationClassifier import LocalizationClassifier

#Hyperparameter Analysis
for radius in LocalizationClassifier.analysis_hyperparameters()['radius']:
    for nofframes in LocalizationClassifier.analysis_hyperparameters()['nofframes']:
        for batch_size in LocalizationClassifier.analysis_hyperparameters()['batch_size']:
            classifier = LocalizationClassifier(10,10)

            classifier.hyperparameters['radius'] = radius
            classifier.hyperparameters['nofframes'] = nofframes
            classifier.hyperparameters['batch_size'] = batch_size

            classifier.fit_with_datasets_from_path('./datasets_shuffled/train')
            classifier.save_model()
            classifier.test_with_datasets_from_path('./datasets_shuffled/test', apply_threshold=True, save_result=True, save_predictions=True)
