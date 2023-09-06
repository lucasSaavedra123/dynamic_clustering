import shutil
import os
from random import shuffle
import pandas as pd
from collections import Counter
from dynamic_clustering.CONSTANTS import CLUSTERIZED_COLUMN_NAME

shuffled_to_dir = 'datasets_shuffled'

files = []
origin_datasets_paths = ['./datasets', './datasets_without_clustering', './datasets_without_new_clusters']

for original_datasets_path in origin_datasets_paths:
    files += [os.path.join(original_datasets_path, file) for file in os.listdir(original_datasets_path) if '.csv' in file]

shuffle(files)

number_of_files = len(files)
number_of_training_files = int(number_of_files * 0.8)

number_of_clusterized_localizations = 0
number_of_non_clusterized_localizations = 0

for new_dataset_number, file in enumerate(files[:number_of_training_files]):
    shutil.copy(file, f'./{shuffled_to_dir}/train/{new_dataset_number}_smlm_dataset.csv')
    dataset = pd.read_csv(file)
    counter = Counter(dataset[CLUSTERIZED_COLUMN_NAME].values)
    number_of_clusterized_localizations += counter[1]
    number_of_non_clusterized_localizations += counter[0]

print("Training:")
print("Clusterized Localizations:", number_of_clusterized_localizations/(number_of_clusterized_localizations+number_of_non_clusterized_localizations))
print("Non Clusterized Localizations:", number_of_non_clusterized_localizations/(number_of_clusterized_localizations+number_of_non_clusterized_localizations))

number_of_clusterized_localizations = 0
number_of_non_clusterized_localizations = 0

for new_dataset_number, file in enumerate(files[number_of_training_files:]):
    shutil.copy(file, f'./{shuffled_to_dir}/test/{new_dataset_number}_smlm_dataset.csv')
    dataset = pd.read_csv(file)
    counter = Counter(dataset[CLUSTERIZED_COLUMN_NAME].values)
    number_of_clusterized_localizations += counter[1]
    number_of_non_clusterized_localizations += counter[0]

print("Testing:")
print("Clusterized Localizations:", number_of_clusterized_localizations/(number_of_clusterized_localizations+number_of_non_clusterized_localizations))
print("Non Clusterized Localizations:", number_of_non_clusterized_localizations/(number_of_clusterized_localizations+number_of_non_clusterized_localizations))
