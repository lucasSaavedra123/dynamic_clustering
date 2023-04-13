from collections import Counter
import os
import tqdm
import pickle
import json

from deeptrack.models.gnns.augmentations import NodeDropout
from deeptrack.models.gnns.generators import ContinuousGraphGenerator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import deeptrack as dt
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import ghostml

from CONSTANTS import *


class ClusterEdgeRemover():
    @classmethod
    def default_hyperparameters(cls):
        return {
            "learning_rate": 0.001,
            "radius": 0.05,
            "nofframes": 50, #20
            "partition_size": 50000,
            "epochs": 25,
            "batch_size": 1,
        }

    @classmethod
    def analysis_hyperparameters(cls):
        return {
            #"learning_rate": [0.1, 0.01, 0.001],
            #"radius": [0.05, 0.1, 0.25],
            #"nofframes": [3, 5, 7, 9, 11],
            #"partition_size": [25, 50, 75, 100],
            "batch_size": [1,2,4]
        }

    def __init__(self, height=10, width=10):
        self._output_type = "edges"

        self.magik_architecture = None
        self.threshold = 0.5

        self.hyperparameters = self.__class__.default_hyperparameters()
        self.height = height
        self.width = width

    @property
    def node_features(self):
        return [MAGIK_X_POSITION_COLUMN_NAME, MAGIK_Y_POSITION_COLUMN_NAME, TIME_COLUMN_NAME]

    @property
    def edge_features(self):
        return ["distance"]

    def build_network(self):
        self.magik_architecture = dt.models.gnns.MAGIK(
            dense_layer_dimensions=(64, 96,),
            base_layer_dimensions=(96, 96, 96),
            number_of_node_features=len(self.node_features),
            number_of_edge_features=len(self.edge_features),
            number_of_edge_outputs=1,
            node_output_activation="sigmoid",
            output_type=self._output_type,
        )

        self.magik_architecture.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.hyperparameters['learning_rate']),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )

        self.magik_architecture.summary()

    def transform_smlm_dataset_to_magik_dataframe(self, smlm_dataframe, set_number=0):
        smlm_dataframe = smlm_dataframe.rename(columns={
            X_POSITION_COLUMN_NAME: MAGIK_X_POSITION_COLUMN_NAME,
            Y_POSITION_COLUMN_NAME: MAGIK_Y_POSITION_COLUMN_NAME,
            CLUSTER_ID_COLUMN_NAME: MAGIK_LABEL_COLUMN_NAME,
            CLUSTER_ID_COLUMN_NAME+"_predicted": MAGIK_LABEL_COLUMN_NAME_PREDICTED,
        })

        smlm_dataframe['original_index_for_recovery'] = smlm_dataframe.index

        if 'clusterized_predicted' in smlm_dataframe.columns:
            smlm_dataframe = smlm_dataframe[smlm_dataframe['clusterized_predicted'] == 1]
        else:
            smlm_dataframe = smlm_dataframe[smlm_dataframe['clusterized'] == 1]

        smlm_dataframe = smlm_dataframe.drop([CLUSTERIZED_COLUMN_NAME, CLUSTERIZED_COLUMN_NAME+'_predicted', PARTICLE_ID_COLUMN_NAME, "Unnamed: 0"], axis=1, errors="ignore")
        smlm_dataframe.loc[:, smlm_dataframe.columns.str.contains(MAGIK_POSITION_COLUMN_NAME)] = (smlm_dataframe.loc[:, smlm_dataframe.columns.str.contains(MAGIK_POSITION_COLUMN_NAME)] / np.array([self.width, self.height]))
        
        smlm_dataframe[TIME_COLUMN_NAME] = smlm_dataframe[TIME_COLUMN_NAME] / smlm_dataframe[TIME_COLUMN_NAME].abs().max()
        smlm_dataframe[MAGIK_DATASET_COLUMN_NAME] = set_number
        smlm_dataframe[MAGIK_LABEL_COLUMN_NAME] = smlm_dataframe[MAGIK_LABEL_COLUMN_NAME].astype(int)

        return smlm_dataframe.reset_index(drop=True)

    def transform_magik_dataframe_to_smlm_dataset(self, magik_dataframe):
        magik_dataframe = magik_dataframe.rename(columns={
            MAGIK_X_POSITION_COLUMN_NAME: X_POSITION_COLUMN_NAME,
            MAGIK_Y_POSITION_COLUMN_NAME: Y_POSITION_COLUMN_NAME,
            MAGIK_LABEL_COLUMN_NAME: CLUSTER_ID_COLUMN_NAME,
            MAGIK_LABEL_COLUMN_NAME_PREDICTED: CLUSTER_ID_COLUMN_NAME+"_predicted",
        })

        magik_dataframe.loc[:, magik_dataframe.columns.str.contains(X_POSITION_COLUMN_NAME)] = (magik_dataframe.loc[:, magik_dataframe.columns.str.contains(X_POSITION_COLUMN_NAME)] * np.array([self.width]))
        magik_dataframe.loc[:, magik_dataframe.columns.str.contains(Y_POSITION_COLUMN_NAME)] = (magik_dataframe.loc[:, magik_dataframe.columns.str.contains(Y_POSITION_COLUMN_NAME)] * np.array([self.height]))

        magik_dataframe = magik_dataframe.drop(MAGIK_DATASET_COLUMN_NAME, axis=1)

        return magik_dataframe.reset_index(drop=True)

    def get_dataset_from_path(self, path, set_number=0):
        return self.transform_smlm_dataset_to_magik_dataframe(pd.read_csv(path), set_number=set_number)

    def get_dataset_file_paths_from(self, path):
        return [os.path.join(path, file_name) for file_name in os.listdir(path) if file_name.endswith(".csv") and len(file_name.split('.'))==2]

    def get_datasets_from_path(self, path):
        """
        This method is different of the Localization Classifier method
        because there are some datasets for testing that have no clusters
        """
        full_dataset = pd.DataFrame({})
        set_index = 0

        for csv_file_path in self.get_dataset_file_paths_from(path):
            set_dataframe = self.get_dataset_from_path(csv_file_path, set_number=set_index)

            if not set_dataframe.empty:
                full_dataset = full_dataset.append(set_dataframe)
                set_index += 1

        return full_dataset.reset_index(drop=True)

    def predict(self, magik_dataset, apply_threshold=True, verbose=True):

        magik_dataset = magik_dataset.copy()

        grapht, original_index = self.build_graph(magik_dataset, for_predict=True)

        v = [
            grapht[0][0].reshape(1, grapht[0][0].shape[0], grapht[0][0].shape[1]),
            grapht[0][1].reshape(1, grapht[0][1].shape[0], grapht[0][1].shape[1]),
            grapht[0][2].reshape(1, grapht[0][2].shape[0], grapht[0][2].shape[1]),
            grapht[0][3].reshape(1, grapht[0][3].shape[0], grapht[0][3].shape[1]),
        ]

        with tf.device('/cpu:0'):
            if apply_threshold:
                predictions = (self.magik_architecture(v).numpy() > self.threshold)[0, ...]
            else:
                predictions = (self.magik_architecture(v).numpy())[0, ...]
        
        if apply_threshold:
            return predictions, grapht[1][1]

        edges_to_remove = np.where(predictions == 0)[0]
        remaining_edges_keep = np.delete(grapht[0][2], edges_to_remove, axis=0)

        cluster_sets = []

        for i in range(len(remaining_edges_keep)):
            cluster_assigned = False
            if len(cluster_sets) == 0:
                cluster_sets.append(set([remaining_edges_keep[i][0], remaining_edges_keep[i][1]]))
            else:
                for index, s in enumerate(cluster_sets):
                    if remaining_edges_keep[i][0] in s or remaining_edges_keep[i][1] in s:
                        s.add(remaining_edges_keep[i][0])
                        s.add(remaining_edges_keep[i][1])
                        cluster_assigned = True
                        break

                if not cluster_assigned:
                    cluster_sets.append(set([remaining_edges_keep[i][0], remaining_edges_keep[i][1]]))

        magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED] = 0

        for index, a_set in enumerate(cluster_sets):
            for value in a_set:
                magik_dataset.loc[original_index[value], MAGIK_LABEL_COLUMN_NAME_PREDICTED] = index + 1
        
        #magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED] = magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED].astype(int)

        if MAGIK_LABEL_COLUMN_NAME in magik_dataset.columns:
            magik_dataset[MAGIK_LABEL_COLUMN_NAME] = magik_dataset[MAGIK_LABEL_COLUMN_NAME].astype(int)

        return magik_dataset

    def build_graph(self, full_nodes_dataset, verbose=True, for_predict=False):
        edges_dataframe = pd.DataFrame({
            "distance": [],
            "index_1": [],
            "index_2": [],
            "set": [],
            "same_cluster": []
        })

        """
        #We iterate on all the datasets and we extract all the edges.
        sets = np.unique(full_nodes_dataset[MAGIK_DATASET_COLUMN_NAME])

        if verbose:
            iterator = tqdm.tqdm(sets)
        else:
            iterator = sets

        clusters_extracted_from_dbscan = pd.DataFrame({})

        global_sub_cluster_id = 0

        dbscan_result_by_subcluster_id = {}

        for setid in iterator:
            df_set = full_nodes_dataset[full_nodes_dataset[MAGIK_DATASET_COLUMN_NAME] == setid].copy().reset_index(drop=True)
            eps_values = np.linspace(0, 1, 100)

            results = {}
            dbscan_results = {}

            for eps_value in eps_values:
                if eps_value != 0:
                    dbscan_result = DBSCAN(eps=eps_value, min_samples=1).fit_predict(df_set[[MAGIK_X_POSITION_COLUMN_NAME, MAGIK_Y_POSITION_COLUMN_NAME]])

                    if len(set(dbscan_result)) == 1:
                        break

                    if -1 not in dbscan_result:
                        results[eps_value] = silhouette_score(df_set[[MAGIK_X_POSITION_COLUMN_NAME, MAGIK_Y_POSITION_COLUMN_NAME]], dbscan_result)
                        dbscan_results[eps_value] = dbscan_result

            selected_eps_value = max(results, key=results.get)
            dbscan_result = dbscan_results[selected_eps_value]

            for sub_cluster_id in set(dbscan_result):
                # remove excess frames
                localizations_in_subcluster = np.where(dbscan_result == sub_cluster_id)
                cluster_df = df_set.loc[localizations_in_subcluster].copy()
                count = Counter(cluster_df[MAGIK_LABEL_COLUMN_NAME])
                if for_predict or (len(cluster_df[MAGIK_LABEL_COLUMN_NAME].unique()) > 1 and (np.array([count[value] for value in count]) > 10).all()):
                    cluster_df[MAGIK_DATASET_COLUMN_NAME] = global_sub_cluster_id
                    dbscan_result_by_subcluster_id[global_sub_cluster_id] = selected_eps_value
                    global_sub_cluster_id += 1
                    clusters_extracted_from_dbscan = clusters_extracted_from_dbscan.append(cluster_df, ignore_index=(not for_predict))

        original_index = clusters_extracted_from_dbscan.index.to_list()
        clusters_extracted_from_dbscan = clusters_extracted_from_dbscan.reset_index(drop=True)
        """

        clusters_extracted_from_dbscan = full_nodes_dataset.reset_index(drop=True)
        clusters_extracted_from_dbscan['index'] = clusters_extracted_from_dbscan.index

        #We iterate on all the datasets and we extract all the edges.
        sets = np.unique(clusters_extracted_from_dbscan[MAGIK_DATASET_COLUMN_NAME])

        if verbose:
            iterator = tqdm.tqdm(sets)
        else:
            iterator = sets
        
        for setid in iterator:
            new_edges_dataframe = pd.DataFrame({'index_1': [], 'index_2': [],'distance': [], 'same_cluster': []})
            df_window = clusters_extracted_from_dbscan[clusters_extracted_from_dbscan[MAGIK_DATASET_COLUMN_NAME] == setid].copy().reset_index(drop=True)
            simplices = Delaunay(df_window[[MAGIK_X_POSITION_COLUMN_NAME, MAGIK_Y_POSITION_COLUMN_NAME, TIME_COLUMN_NAME]].values).simplices

            def less_first(a, b):
                return [a,b] if a < b else [b,a]

            list_of_edges = []

            for triangle in simplices:
                for e1, e2 in [[0,1],[1,2],[2,0]]: # for all edges of triangle
                    list_of_edges.append(less_first(triangle[e1],triangle[e2])) # always lesser index first

            new_index_to_old_index = {new_index:df_window.loc[new_index, 'index'] for new_index in df_window.index.values}
            list_of_edges = np.vectorize(new_index_to_old_index.get)(list_of_edges)
            list_of_edges = np.unique(list_of_edges, axis=0).tolist() # remove duplicates

            simplified_cross = pd.DataFrame({
                f"{MAGIK_X_POSITION_COLUMN_NAME}_x": [],
                f"{MAGIK_X_POSITION_COLUMN_NAME}_y": [],
                f"{MAGIK_Y_POSITION_COLUMN_NAME}_x": [],
                f"{MAGIK_Y_POSITION_COLUMN_NAME}_y": [],
                'index_x': [],
                'index_y': [],
                MAGIK_LABEL_COLUMN_NAME+"_x": [],
                MAGIK_LABEL_COLUMN_NAME+"_y": [],
                TIME_COLUMN_NAME+"_x": [],
                TIME_COLUMN_NAME+"_y": []
            })

            for edge in list_of_edges:
                x_index = df_window["index"] == edge[0]
                y_index = df_window["index"] == edge[1]

                simplified_cross = simplified_cross.append(pd.DataFrame({
                    f"{MAGIK_X_POSITION_COLUMN_NAME}_x": [df_window[x_index][f"{MAGIK_X_POSITION_COLUMN_NAME}"].values[0]],
                    f"{MAGIK_X_POSITION_COLUMN_NAME}_y": [df_window[y_index][f"{MAGIK_X_POSITION_COLUMN_NAME}"].values[0]],
                    f"{MAGIK_Y_POSITION_COLUMN_NAME}_x": [df_window[x_index][f"{MAGIK_Y_POSITION_COLUMN_NAME}"].values[0]],
                    f"{MAGIK_Y_POSITION_COLUMN_NAME}_y": [df_window[y_index][f"{MAGIK_Y_POSITION_COLUMN_NAME}"].values[0]],
                    'index_x': [edge[0]],
                    'index_y': [edge[1]],
                    MAGIK_LABEL_COLUMN_NAME+"_x": [df_window[x_index][f"{MAGIK_LABEL_COLUMN_NAME}"].values[0]],
                    MAGIK_LABEL_COLUMN_NAME+"_y": [df_window[y_index][f"{MAGIK_LABEL_COLUMN_NAME}"].values[0]],
                    TIME_COLUMN_NAME+"_x": [df_window[x_index][f"{TIME_COLUMN_NAME}"].values[0]],
                    TIME_COLUMN_NAME+"_y": [df_window[y_index][f"{TIME_COLUMN_NAME}"].values[0]],
                }), ignore_index=True)

            df_window = simplified_cross.copy()
            df_window = df_window[df_window['index_x'] != df_window['index_y']]
            df_window['distance-x'] = df_window[f"{MAGIK_X_POSITION_COLUMN_NAME}_x"] - df_window[f"{MAGIK_X_POSITION_COLUMN_NAME}_y"]
            df_window['distance-y'] = df_window[f"{MAGIK_Y_POSITION_COLUMN_NAME}_x"] - df_window[f"{MAGIK_Y_POSITION_COLUMN_NAME}_y"]
            df_window['distance'] = ((df_window['distance-x']**2) + (df_window['distance-y']**2))**(1/2)
            df_window['same_cluster'] = (df_window[MAGIK_LABEL_COLUMN_NAME+"_x"] == df_window[MAGIK_LABEL_COLUMN_NAME+"_y"])
            #df_window = df_window[df_window['distance'] < self.hyperparameters['radius']]

            if for_predict or not df_window['same_cluster'].all():
                edges = [sorted(edge) for edge in df_window[["index_x", "index_y"]].values.tolist()]

                new_edges_dataframe = new_edges_dataframe.append(pd.DataFrame({
                    'index_1': [edge[0] for edge in edges],
                    'index_2': [edge[1] for edge in edges],
                    'distance': [value[0] for value in df_window[["distance"]].values.tolist()],
                    'same_cluster': [value[0] for value in df_window[["same_cluster"]].values.tolist()],
                }), ignore_index=True)

            new_edges_dataframe = new_edges_dataframe.drop_duplicates()
            new_edges_dataframe['set'] = setid
            edges_dataframe = edges_dataframe.append(new_edges_dataframe, ignore_index=True)

        edgefeatures = edges_dataframe[self.edge_features].to_numpy()
        sparseadjmtx = edges_dataframe[["index_1", "index_2"]].to_numpy().astype(int)
        nodefeatures = clusters_extracted_from_dbscan[self.node_features].to_numpy()

        edgeweights = np.ones(sparseadjmtx.shape[0])
        edgeweights = np.stack((np.arange(0, edgeweights.shape[0]), edgeweights), axis=1)

        nfsolution = np.zeros((len(nodefeatures), 1))
        efsolution = edges_dataframe[['same_cluster']].to_numpy().astype(int)

        nodesets = clusters_extracted_from_dbscan[[MAGIK_DATASET_COLUMN_NAME]].to_numpy().astype(int)
        edgesets = edges_dataframe[[MAGIK_DATASET_COLUMN_NAME]].to_numpy().astype(int)
        framesets = clusters_extracted_from_dbscan[[FRAME_COLUMN_NAME]].to_numpy().astype(int)

        global_property = np.zeros(np.unique(clusters_extracted_from_dbscan[MAGIK_DATASET_COLUMN_NAME]).shape[0])

        if for_predict:
            return (
                (nodefeatures, edgefeatures, sparseadjmtx, edgeweights),
                (nfsolution, efsolution, global_property),
                (nodesets, edgesets, framesets)
            ), original_index
        else:
            return (
                (nodefeatures, edgefeatures, sparseadjmtx, edgeweights),
                (nfsolution, efsolution, global_property),
                (nodesets, edgesets, framesets)
            )            

    @property
    def train_full_graph_file_name(self):
        return f"edge_classifier.tmp"

    @property
    def model_file_name(self):
        return f"edge_classifier_batch_size_{self.hyperparameters['batch_size']}.h5"

    @property
    def threshold_file_name(self):
        return f"edge_classifier_batch_size_{self.hyperparameters['batch_size']}.bin"

    @property
    def predictions_file_name(self):
        return f"edge_classifier_batch_size_{self.hyperparameters['batch_size']}.csv"
    
    @property
    def history_training_info_file_name(self):
        return f"edge_classifier_batch_size_{self.hyperparameters['batch_size']}.json"

    def test_with_datasets_from_path(self, path, plot=False, apply_threshold=True, save_result=False, save_predictions=False, verbose=True):
        true = []
        pred = []

        iterator = tqdm.tqdm(self.get_dataset_file_paths_from(path)) if verbose else self.get_dataset_file_paths_from(path)

        for csv_file_name in iterator:
            r = self.predict(self.get_dataset_from_path(csv_file_name), apply_threshold=apply_threshold, verbose=False)

            if save_predictions:
                r.to_csv(csv_file_name+f"_predicted_with_batch_size_{self.hyperparameters['batch_size']}_{self.hyperparameters['radius']}_nofframes_{self.hyperparameters['nofframes']}_partition_{self.hyperparameters['partition_size']}.csv", index=False)

            true += r[MAGIK_LABEL_COLUMN_NAME].values.tolist()
            pred += r[MAGIK_LABEL_COLUMN_NAME_PREDICTED].values.tolist()

        if save_result:
            pd.DataFrame({
                'true': true,
                'pred': pred
            }).to_csv(self.predictions_file_name, index=False)

        if plot:
            raise NotImplementedError("Plotting during testing is not implemented yet for Cluster Edge Remover")

        return true, pred

    def fit_with_datasets_from_path(self, path):
        if os.path.exists(self.train_full_graph_file_name):
            fileObj = open(self.train_full_graph_file_name, 'rb')
            train_full_graph = pickle.load(fileObj)
            fileObj.close()
        else:
            train_full_graph = self.build_graph(self.get_datasets_from_path(path))
            fileObj = open(self.train_full_graph_file_name, 'wb')
            pickle.dump(train_full_graph, fileObj)
            fileObj.close()

        self.build_network()

        if self.load_keras_model() is None:

            def CustomGetSubSet():
                """
                Returns a function that takes a graph and returns a
                random subset of the graph.
                """

                def inner(data):
                    graph, labels, sets = data

                    retry = True

                    while retry:
                        randset = np.random.randint(np.max(np.array(sets[0])[:,0]) + 1)

                        nodeidxs = np.where(sets[0][:, 0] == randset)[0]
                        edgeidxs = np.where(sets[1][:, 0] == randset)[0]

                        min_node = np.min(nodeidxs)

                        node_features = graph[0][nodeidxs]
                        edge_features = graph[1][edgeidxs]
                        edge_connections = graph[2][edgeidxs] - min_node

                        weights = graph[3][edgeidxs]

                        node_labels = labels[0][nodeidxs]
                        edge_labels = labels[1][edgeidxs]
                        glob_labels = labels[2][randset]

                        """
                        node_sets = sets[0][nodeidxs]
                        edge_sets = sets[1][edgeidxs]

                        node_sets[:,0] = 0
                        edge_sets[:,0] = 0
                        """

                        frame_sets = sets[2][nodeidxs]
                        count = Counter(np.array(edge_labels)[:,0])
                        retry = len(count) == 0 or count[0] == 0 or count[1] == 0

                    return (node_features, edge_features, edge_connections, weights), (
                        node_labels,
                        edge_labels,
                        glob_labels,
                    )#, (frame_sets) #, (node_sets, edge_sets, frame_sets)

                return inner

            def CustomGetSubGraph():
                def inner(data):
                    graph, labels = data

                    min_num_nodes = 2500
                    max_num_nodes = 5000

                    retry = True

                    while retry:

                        num_nodes = np.random.randint(min_num_nodes, max_num_nodes+1)

                        node_start = np.random.randint(max(len(graph[0]) - num_nodes, 1))

                        edge_connects_removed_node = np.any((graph[2] < node_start) | (graph[2] >= node_start + num_nodes),axis=-1)

                        node_features = graph[0][node_start : node_start + num_nodes]
                        edge_features = graph[1][~edge_connects_removed_node]
                        edge_connections = graph[2][~edge_connects_removed_node] - node_start
                        weights = graph[3][~edge_connects_removed_node]

                        node_labels = labels[0][node_start : node_start + num_nodes]
                        edge_labels = labels[1][~edge_connects_removed_node]
                        global_labels = labels[2]

                        count = Counter(np.array(edge_labels)[:,0])
                        retry = len(np.array(edge_labels)) == 0 or count[0] == 0 or count[1] == 0

                    return (node_features, edge_features, edge_connections, weights), (
                        node_labels,
                        edge_labels,
                        global_labels,
                    )

                return inner

            def CustomDatasetBalancing():
                def inner(data):
                    graph, labels = data

                    number_of_same_cluster_edges = np.sum(np.array((labels[1][:, 0] == 1)) * 1)
                    number_of_non_same_cluster_edges = np.sum(np.array(labels[1][:, 0] == 0) * 1)

                    if number_of_same_cluster_edges > number_of_non_same_cluster_edges:
                        edgeidxs = np.array(np.where(labels[1][:, 0] == 1)[0])

                        number_of_edges_to_select = number_of_non_same_cluster_edges

                        edges_to_select = np.random.choice(edgeidxs, size=number_of_edges_to_select, replace=False)
                        edges_to_select = np.append(edges_to_select, np.array(np.where(labels[1][:, 0] == 0)[0]))
                        nodes_to_select = np.unique(np.array(graph[2][edges_to_select]))

                        id_to_new_id = {}

                        for index, value in enumerate(nodes_to_select):
                            id_to_new_id[value] = index

                        node_features = graph[0][nodes_to_select]
                        edge_features = graph[1][edges_to_select]
                        edge_connections = graph[2][edges_to_select]

                        edge_connections = np.vectorize(id_to_new_id.get)(edge_connections)

                        weights = graph[3][edges_to_select]

                        node_labels = labels[0][nodes_to_select]
                        edge_labels = labels[1][edges_to_select]
                        global_labels = labels[2]

                        return (node_features, edge_features, edge_connections, weights), (
                            node_labels,
                            edge_labels,
                            global_labels,
                        )
                    else:
                        return graph, labels

                return inner

            def CustomAugmentCentroids(rotate, flip_x, flip_y):
                def inner(data):
                    graph, labels = data

                    centroids = graph[0][:, :2]

                    centroids = centroids - 0.5
                    centroids_x = (
                        centroids[:, 0] * np.cos(rotate)
                        + centroids[:, 1] * np.sin(rotate)
                    )
                    centroids_y = (
                        centroids[:, 1] * np.cos(rotate)
                        - centroids[:, 0] * np.sin(rotate)
                    )
                    if flip_x:
                        centroids_x *= -1
                    if flip_y:
                        centroids_y *= -1

                    node_features = np.array(graph[0])
                    node_features[:, 0] = centroids_x + 0.5
                    node_features[:, 1] = centroids_y + 0.5

                    return (node_features, *graph[1:]), labels

                return inner

            def CustomGetFeature(full_graph, **kwargs):
                return (
                    dt.Value(full_graph)
                    >> dt.Lambda(CustomGetSubSet)
                    >> dt.Lambda(CustomGetSubGraph)
                    >> dt.Lambda(CustomDatasetBalancing)
                    >> dt.Lambda(
                        CustomAugmentCentroids,
                        rotate=lambda: np.random.rand() * 2 * np.pi,
                        translate=lambda: np.random.randn(2) * 0,
                        flip_x=lambda: np.random.randint(2),
                        flip_y=lambda: np.random.randint(2)
                    )
                    >> dt.Lambda(NodeDropout, dropout_rate=0.00)
                )

            magik_variables = dt.DummyFeature(
                radius=self.hyperparameters["radius"],
                output_type=self._output_type,
                nofframes=self.hyperparameters["nofframes"],  # time window to associate nodes (in frames)
            )

            args = {
                "batch_function": lambda graph: graph[0],
                "label_function": lambda graph: graph[1],
                "min_data_size": 512,
                "max_data_size": 513,
                "batch_size": self.hyperparameters["batch_size"],
                "use_multi_inputs": False,
                **magik_variables.properties(),
            }

            generator = ContinuousGraphGenerator(CustomGetFeature(train_full_graph, **magik_variables.properties()), **args)

            with generator:
                self.magik_architecture.fit(generator, epochs=self.hyperparameters["epochs"])

            del generator
        
        del train_full_graph

        if self.load_threshold() is None:
            print("Running Ghost...")
            true = []
            pred = []

            true, pred = self.test_with_datasets_from_path(path, apply_threshold=False, save_result=False, verbose=True)

            count = Counter(true)
            positive_is_majority = count[1] > count[0]

            if positive_is_majority:
                true = 1 - np.array(true)
                pred = 1 - np.array(pred)

            thresholds = np.round(np.arange(0.05,0.95,0.025), 3)

            self.threshold = ghostml.optimize_threshold_from_predictions(true, pred, thresholds, ThOpt_metrics = 'ROC')

            if positive_is_majority:
                self.threshold = 1 - self.threshold

    def plot_confusion_matrix(self, ground_truth, Y_predicted, normalized=True):
        confusion_mat = confusion_matrix(y_true=ground_truth, y_pred=Y_predicted)

        if normalized:
            confusion_mat = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]

        labels = ["Non-Clusterized", "Clusterized"]

        confusion_matrix_dataframe = pd.DataFrame(data=confusion_mat, index=labels, columns=labels)
        sns.set(font_scale=1.5)
        color_map = sns.color_palette(palette="Blues", n_colors=7)
        sns.heatmap(data=confusion_matrix_dataframe, annot=True, annot_kws={"size": 15}, cmap=color_map)

        plt.title(f'Confusion Matrix')
        plt.rcParams.update({'font.size': 15})
        plt.ylabel("Ground truth", fontsize=15)
        plt.xlabel("Predicted label", fontsize=15)
        plt.show()

    def save_threshold(self):
        with open(self.threshold_file_name, "w") as threshold_file:
            threshold_file.write(str(self.threshold))

    def save_history_training_info(self):
        with open(self.history_training_info_file_name, "w") as json_file:
            json.dump(self.history_training_info, json_file)

    def save_model(self):
        self.save_keras_model()
        self.save_threshold()
        self.save_history_training_info()

    def save_keras_model(self):
        self.magik_architecture.save_weights(self.model_file_name)

    def load_threshold(self):
        try:
            with open(self.threshold_file_name, "r") as threshold_file:
                self.threshold = float(threshold_file.read())
        except FileNotFoundError:
            return None

        return self.threshold

    def load_history_training_info(self):
        try:
            with open(self.history_training_info_file_name, "r") as json_file:
                self.history_training_info = json.load(json_file)
        except FileNotFoundError:
            return None

        return self.history_training_info

    def load_keras_model(self):
        try:
            self.build_network()
            self.magik_architecture.load_weights(self.model_file_name)
        except FileNotFoundError:
            return None

        return self.magik_architecture

    def load_model(self):
        self.load_keras_model()
        self.load_threshold()
        self.load_history_training_info()
