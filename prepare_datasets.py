import shutil
import os
from random import shuffle
import pandas as pd

shuffled_to_dir = 'datasets_shuffled'

files = [f'./datasets/{file}' for file in os.listdir(f'./datasets') if '.csv' in file]
files += [f'./datasets_without_clustering/{file}' for file in os.listdir(f'./datasets_without_clustering') if '.csv' in file]
files += [f'./datasets_without_new_clusters/{file}' for file in os.listdir(f'./datasets_without_new_clusters') if '.csv' in file]
shuffle(files)

for new_dataset_number, file in enumerate(files):
    shutil.copy(file, f'./{shuffled_to_dir}/{new_dataset_number}_smlm_dataset.csv')
    a = pd.read_csv(f'./{shuffled_to_dir}/{new_dataset_number}_smlm_dataset.csv')

    number_of_clusterized += len(a[a['clusterized'] == 1])
    number_of_non_clusterized += len(a[a['clusterized'] == 0])
