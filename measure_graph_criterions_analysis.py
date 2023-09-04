import os
from utils import *
from collections import Counter
import time
import pickle

import pandas as pd
import tqdm
from sklearn.metrics import adjusted_rand_score

from ClusterDetector import ClusterDetector
from LocalizationClassifier import LocalizationClassifier
from CONSTANTS import *


RS = [0.01,0.025,0.05,0.075,0.1]
WS = [3,5,7,11]
TEST_DATASETS_PATH = "./datasets_shuffled/test"

cluster_detector = ClusterDetector(10,10)
localization_classifier = LocalizationClassifier(10,10)

if not (
    os.path.exists(f'delaunay_criterion_gcu.txt') and
    os.path.exists(f'delaunay_criterion_negative_and_positive_edges.txt') and
    os.path.exists(f'delaunay_criterion_times.txt') and
    os.path.exists(f'delaunay_criterion_memory.txt') and
    os.path.exists(f'delaunay_criterion_edges.txt')
    ):

    delaunay_criterion_gcu = []
    delaunay_criterion_negative_and_positive_edges = []
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

        magik_dataset = cluster_detector.transform_smlm_dataset_to_magik_dataframe(pd.read_csv(dataset_file_path), ignore_non_clustered_localizations=False)

        if len(magik_dataset[magik_dataset[CLUSTERIZED_COLUMN_NAME] == 1]) != 0:
            grapht, real_edges_weights = cluster_detector.build_graph(magik_dataset, verbose=False, return_real_edges_weights=True)
            prediction = cluster_detector.predict(magik_dataset, verbose=False, suppose_perfect_classification=True, grapht=grapht, real_edges_weights=real_edges_weights)

            counting = Counter(grapht[1][1].flatten().tolist())
            delaunay_criterion_negative_and_positive_edges.append([counting[0], counting[1]])

            gcu = adjusted_rand_score(prediction[MAGIK_LABEL_COLUMN_NAME], prediction[MAGIK_LABEL_COLUMN_NAME_PREDICTED])
            delaunay_criterion_gcu.append(gcu)

    save_numbers_in_file(f'delaunay_criterion_gcu.txt', delaunay_criterion_gcu)
    save_numbers_in_file(f'delaunay_criterion_times.txt', delaunay_criterion_times)
    save_numbers_in_file(f'delaunay_criterion_memory.txt', delaunay_criterion_memory)
    save_numbers_in_file(f'delaunay_criterion_edges.txt', delaunay_criterion_edges)
    save_number_lists_in_file(f'delaunay_criterion_negative_and_positive_edges.txt', delaunay_criterion_negative_and_positive_edges)

for r in RS:
    for w in WS:
        print(r,w)
        if not(
            os.path.exists(f'spatiotemporal_criterion_r_{r}_w_{w}_gcu.txt') and
            os.path.exists(f'spatiotemporal_criterion_r_{r}_w_{w}_negative_and_positive_edges.txt') and
            os.path.exists(f'spatiotemporal_criterion_r_{r}_w_{w}_times.txt') and
            os.path.exists(f'spatiotemporal_criterion_r_{r}_w_{w}_memory.txt') and
            os.path.exists(f'spatiotemporal_criterion_r_{r}_w_{w}_edges')
            ):

            spatiotemporal_criterion_gcu = []
            spatiotemporal_criterion_negative_and_positive_edges = []
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

                magik_dataset = cluster_detector.transform_smlm_dataset_to_magik_dataframe(pd.read_csv(dataset_file_path), ignore_non_clustered_localizations=False)

                if len(magik_dataset[magik_dataset[CLUSTERIZED_COLUMN_NAME] == 1]) != 0:
                    grapht, real_edges_weights = build_graph_with_spatio_temporal_criterion(magik_dataset, r, w, ["distance"], [MAGIK_X_POSITION_COLUMN_NAME, MAGIK_Y_POSITION_COLUMN_NAME], return_real_edges_weights=True)
                    prediction = cluster_detector.predict(magik_dataset, verbose=False, suppose_perfect_classification=True, grapht=grapht, real_edges_weights=real_edges_weights)

                    counting = Counter(grapht[1][1].flatten().tolist())
                    spatiotemporal_criterion_negative_and_positive_edges.append([counting[0], counting[1]])

                    gcu = adjusted_rand_score(prediction[MAGIK_LABEL_COLUMN_NAME], prediction[MAGIK_LABEL_COLUMN_NAME_PREDICTED])
                    spatiotemporal_criterion_gcu.append(gcu)

            save_numbers_in_file(f'spatiotemporal_criterion_r_{r}_w_{w}_gcu.txt', spatiotemporal_criterion_gcu)
            save_numbers_in_file(f'spatiotemporal_criterion_r_{r}_w_{w}_times.txt', spatiotemporal_criterion_times)
            save_numbers_in_file(f'spatiotemporal_criterion_r_{r}_w_{w}_memory.txt', spatiotemporal_criterion_memory)
            save_numbers_in_file(f'spatiotemporal_criterion_r_{r}_w_{w}_edges.txt', spatiotemporal_criterion_edges)
            save_number_lists_in_file(f'spatiotemporal_criterion_r_{r}_w_{w}_negative_and_positive_edges.txt', spatiotemporal_criterion_negative_and_positive_edges)
