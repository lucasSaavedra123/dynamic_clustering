import os
import time
import sys
from utils import *
import pickle

import pandas as pd
import tqdm

from LocalizationClassifier import LocalizationClassifier


localization_classifier = LocalizationClassifier(10,10)

TEST_DATASETS_PATH = "./datasets_shuffled/test"

delaunay_criterion_times = []
spatiotemporal_criterion_times = []

delaunay_criterion_memory = []
spatiotemporal_criterion_memory = []

for dataset_file_path in tqdm.tqdm([os.path.join(TEST_DATASETS_PATH, file) for file in os.listdir(TEST_DATASETS_PATH) if file.endswith('_smlm_dataset.csv')]):
    magik_dataset = localization_classifier.transform_smlm_dataset_to_magik_dataframe(pd.read_csv(dataset_file_path))
    st = time.time()
    graph = localization_classifier.build_graph(magik_dataset, verbose=False)
    et = time.time()

    fileObj = open('tmp.tmp', 'wb')
    pickle.dump(graph, fileObj)
    fileObj.close()

    delaunay_criterion_times.append(et-st)
    delaunay_criterion_memory.append(os.path.getsize('tmp.tmp'))
    os.remove('tmp.tmp')

    magik_dataset = localization_classifier.transform_smlm_dataset_to_magik_dataframe(pd.read_csv(dataset_file_path))
    st = time.time()
    graph = build_graph_with_spatio_temporal_criterion(magik_dataset, 0.01, 5, ["distance"], [MAGIK_X_POSITION_COLUMN_NAME, MAGIK_Y_POSITION_COLUMN_NAME])
    et = time.time()

    fileObj = open('tmp.tmp', 'wb')
    pickle.dump(graph, fileObj)
    fileObj.close()

    spatiotemporal_criterion_times.append(et-st)
    spatiotemporal_criterion_memory.append(os.path.getsize('tmp.tmp'))
    os.remove('tmp.tmp')

save_numbers_in_file(f'delaunay_criterion_times.txt', delaunay_criterion_times)
save_numbers_in_file(f'spatiotemporal_criterion_times.txt', spatiotemporal_criterion_times)

save_numbers_in_file(f'delaunay_criterion_memory.txt', delaunay_criterion_memory)
save_numbers_in_file(f'spatiotemporal_criterion_memory.txt', spatiotemporal_criterion_memory)
