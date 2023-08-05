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

if not (os.path.exists(f'delaunay_criterion_times.txt') and os.path.exists(f'delaunay_criterion_memory.txt') and os.path.exists(f'delaunay_criterion_edges.txt')):
    delaunay_criterion_times = []
    delaunay_criterion_memory = []
    delaunay_criterion_edges = []

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
        delaunay_criterion_edges.append(len(graph[0][1]))

    save_numbers_in_file(f'delaunay_criterion_times.txt', delaunay_criterion_times)
    save_numbers_in_file(f'delaunay_criterion_memory.txt', delaunay_criterion_memory)
    save_numbers_in_file(f'delaunay_criterion_edges.txt', delaunay_criterion_edges)

for r in [0.01,0.025,0.05,0.075,0.1]:
    for w in [3,5,7,11]:
        print(r,w)
        if not(os.path.exists(f'spatiotemporal_criterion_r_{r}_w_{w}_times.txt') and os.path.exists(f'spatiotemporal_criterion_r_{r}_w_{w}_memory.txt') and os.path.exists(f'spatiotemporal_criterion_r_{r}_w_{w}_edges.txt')):
            spatiotemporal_criterion_times = []
            spatiotemporal_criterion_memory = []
            spatiotemporal_criterion_edges = []

            for dataset_file_path in tqdm.tqdm([os.path.join(TEST_DATASETS_PATH, file) for file in os.listdir(TEST_DATASETS_PATH) if file.endswith('_smlm_dataset.csv')]):

                magik_dataset = localization_classifier.transform_smlm_dataset_to_magik_dataframe(pd.read_csv(dataset_file_path))
                st = time.time()
                graph = build_graph_with_spatio_temporal_criterion(magik_dataset, r, w, ["distance"], [MAGIK_X_POSITION_COLUMN_NAME, MAGIK_Y_POSITION_COLUMN_NAME])
                et = time.time()

                fileObj = open('tmp.tmp', 'wb')
                pickle.dump(graph, fileObj)
                fileObj.close()

                spatiotemporal_criterion_times.append(et-st)
                spatiotemporal_criterion_memory.append(os.path.getsize('tmp.tmp'))
                os.remove('tmp.tmp')
                spatiotemporal_criterion_edges.append(len(graph[0][1]))

            save_numbers_in_file(f'spatiotemporal_criterion_r_{r}_w_{w}_times.txt', spatiotemporal_criterion_times)
            save_numbers_in_file(f'spatiotemporal_criterion_r_{r}_w_{w}_memory.txt', spatiotemporal_criterion_memory)
            save_numbers_in_file(f'spatiotemporal_criterion_r_{r}_w_{w}_edges.txt', spatiotemporal_criterion_edges)
