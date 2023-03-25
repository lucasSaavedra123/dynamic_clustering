import shutil
import os
from random import shuffle
import pandas as pd

shuffled_to_dir = 'datasets_shuffled'

files = []
origin_datasets_paths = ['./datasets', './datasets_without_clustering', './datasets_without_new_clusters']

for original_datasets_path in origin_datasets_paths:
    files += [os.path.join(original_datasets_path, file) for file in os.listdir(original_datasets_path) if '.csv' in file]

shuffle(files)

number_of_files = len(files)
number_of_training_files = int(number_of_files * 0.8)

for new_dataset_number, file in enumerate(files[:number_of_training_files]):
    shutil.copy(file, f'./{shuffled_to_dir}/train/{new_dataset_number}_smlm_dataset.csv')

for new_dataset_number, file in enumerate(files[number_of_training_files:]):
    shutil.copy(file, f'./{shuffled_to_dir}/test/{new_dataset_number}_smlm_dataset.csv')
