from CONSTANTS import *
from deeptrack.models.gnns.augmentations import AugmentCentroids, NodeDropout
from deeptrack.models.gnns.generators import GraphExtractor, ContinuousGraphGenerator
import deeptrack as dt
import tensorflow as tf
import numpy as np
import pandas as pd
import math
import os
from scipy.spatial import Delaunay
import logging
import more_itertools as mit
import tqdm
from operator import is_not
from functools import partial
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn import metrics
logging.disable(logging.WARNING)


class SubClusterLinking():
    def __init__(self, height=10, width=10):
        self._output_type = "edges"
        self.magik_architecture = None
        self.hyperparameters = self.__class__.default_hyperparameters()
        self.height = height
        self.width = width

    @classmethod
    def default_hyperparameters(cls):
        return {
            "learning_rate": 0.001,
            "radius": 0.05,
            "nofframes": 50, #20
            "partition_size": 50000,
            "epochs": 25,
            "batch_size": 4,
        }

    @classmethod
    def analysis_hyperparameters(cls):
        return {
            "learning_rate": [0.1, 0.01, 0.001],
            "radius": [0.05, 0.1, 0.25],
            "nofframes": [3, 5, 7, 9, 11],
            "partition_size": [25, 50, 75, 100],
            "batch_size": [1,2,4]
        }

    def build_network(self):
        self.magik_architecture = dt.models.gnns.MAGIK(
            # number of features in each dense encoder layer
            dense_layer_dimensions=(64, 96,),
            # Latent dimension throughout the message passing layers
            base_layer_dimensions=(96, 96, 96),
            number_of_node_features=3,              # Number of node features in the graphs
            number_of_edge_features=1,              # Number of edge features in the graphs
            number_of_edge_outputs=1,               # Number of predicted features
            # Activation function for the output layer
            node_output_activation="sigmoid",
            # Output type. Either "edges", "nodes", or "graph"
            output_type=self._output_type,
        )

        self.magik_architecture.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.hyperparameters['learning_rate']),
            #loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            loss="mse",
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()]
        )

        self.magik_architecture.summary()

    def transform_magik_dataframe_to_smlm_dataset(self, magik_dataframe):
        # normalize centroids between 0 and 1
        magik_dataframe.loc[:, magik_dataframe.columns.str.contains(POSITION_COLUMN_NAME)] = (magik_dataframe.loc[:, magik_dataframe.columns.str.contains(POSITION_COLUMN_NAME)] * np.array([self.width, self.height]))

        magik_dataframe = magik_dataframe.rename(columns={
            f"{POSITION_COLUMN_NAME}-x": X_POSITION_COLUMN_NAME,
            f"{POSITION_COLUMN_NAME}-y": Y_POSITION_COLUMN_NAME,
            LABEL_COLUMN_NAME: CLUSTER_ID_COLUMN_NAME,
            LABEL_COLUMN_NAME+"_predicted": CLUSTER_ID_COLUMN_NAME+"_predicted",
        })

        magik_dataframe = magik_dataframe.drop(DATASET_COLUMN_NAME, axis=1)
        return magik_dataframe

    def transform_smlm_dataset_to_magik_dataframe(self, smlm_dataframe, set_number=0, coming_from_binary_node_classification=False):
        smlm_dataframe = smlm_dataframe.rename(columns={
            X_POSITION_COLUMN_NAME: f"{POSITION_COLUMN_NAME}-x",
            Y_POSITION_COLUMN_NAME: f"{POSITION_COLUMN_NAME}-y",
            CLUSTER_ID_COLUMN_NAME: LABEL_COLUMN_NAME,
        })

        if coming_from_binary_node_classification:    
            smlm_dataframe = smlm_dataframe[smlm_dataframe[CLUSTERIZED_COLUMN_NAME+'_predicted'] == 1]
        else:
            smlm_dataframe = smlm_dataframe[smlm_dataframe[CLUSTERIZED_COLUMN_NAME] == 1]

        if CLUSTERIZED_COLUMN_NAME in smlm_dataframe.columns:
            smlm_dataframe = smlm_dataframe.drop(CLUSTERIZED_COLUMN_NAME, axis=1)
        if CLUSTERIZED_COLUMN_NAME+'_predicted' in smlm_dataframe.columns:
            smlm_dataframe = smlm_dataframe.drop(CLUSTERIZED_COLUMN_NAME+'_predicted', axis=1)
        if PARTICLE_ID_COLUMN_NAME in smlm_dataframe.columns:
            smlm_dataframe = smlm_dataframe.drop(PARTICLE_ID_COLUMN_NAME, axis=1)
        if "Unnamed: 0" in smlm_dataframe.columns:
            smlm_dataframe = smlm_dataframe.drop("Unnamed: 0", axis=1)

        # normalize centroids between 0 and 1
        smlm_dataframe.loc[:, smlm_dataframe.columns.str.contains(POSITION_COLUMN_NAME)] = (smlm_dataframe.loc[:, smlm_dataframe.columns.str.contains(POSITION_COLUMN_NAME)] / np.array([self.width, self.height]))
        smlm_dataframe[TIME_COLUMN_NAME] = smlm_dataframe[TIME_COLUMN_NAME] / smlm_dataframe[TIME_COLUMN_NAME].abs().max()

        smlm_dataframe[DATASET_COLUMN_NAME] = set_number
        smlm_dataframe[LABEL_COLUMN_NAME] = smlm_dataframe[LABEL_COLUMN_NAME].astype(float)

        return smlm_dataframe.reset_index(drop=True)

    def get_dataset_from_path(self, path, set_number=0):
        return self.transform_smlm_dataset_to_magik_dataframe(pd.read_csv(path), set_number=set_number)

    def get_datasets_from_path(self, path):
        file_names = [file_name for file_name in os.listdir(path) if file_name.endswith(".csv")]
        full_dataset = pd.DataFrame({})
        set_index = 0

        for csv_file_name in file_names[0:4]:
            set_dataframe = self.get_dataset_from_path(os.path.join(path, csv_file_name), set_number=set_index)

            if not set_dataframe.empty:
                full_dataset = full_dataset.append(set_dataframe)
                set_index += 1

        return full_dataset.reset_index(drop=True)

    def predict(self, magik_dataset, threshold=0.5):
        magik_dataset = magik_dataset.copy()
        remaining_edges_keep = None

        grapht, original_index = self.build_graph(magik_dataset, for_predict=True)

        for set_index in range(np.max(np.array(grapht[2][0])[:,0])):
            nodeidxs = np.where(grapht[2][0][:, 0] == set_index)[0]
            edgeidxs = np.where(grapht[2][1][:, 0] == set_index)[0]

            node_features = grapht[0][0][nodeidxs]
            edge_features = grapht[0][1][edgeidxs]
            edge_connections = grapht[0][2][edgeidxs]
            edge_weights = grapht[0][3][edgeidxs]

            mapping_old_node_id_to_new_node_id = {old_node_id: new_node_id for new_node_id, old_node_id in enumerate(nodeidxs)}

            def map_node_id(node_id):
                return mapping_old_node_id_to_new_node_id[node_id]

            def inverse_map_node_id(node_id):
                return {v: k for k, v in mapping_old_node_id_to_new_node_id.items()}[node_id]

            edge_connections = np.vectorize(map_node_id)(edge_connections)

            v = [
                node_features.reshape(1, node_features.shape[0], node_features.shape[1]),
                edge_features.reshape(1, edge_features.shape[0], edge_features.shape[1]),
                edge_connections.reshape(1, edge_connections.shape[0], edge_connections.shape[1]),
                edge_weights.reshape(1, edge_weights.shape[0], edge_weights.shape[1]),
            ]

            predictions = (self.magik_architecture(v).numpy() > threshold)[0, ...]
            edges_to_remove = np.where(predictions == 0)[0]
            current_remaining_edges_keep = np.delete(edge_connections, edges_to_remove, axis=0)
            current_remaining_edges_keep = np.vectorize(inverse_map_node_id)(current_remaining_edges_keep)

            if remaining_edges_keep is None:
                remaining_edges_keep = current_remaining_edges_keep
            else:
                remaining_edges_keep = np.append(remaining_edges_keep, current_remaining_edges_keep, axis=0)

        """
        edges_to_remove = np.where(grapht[1][1] == 0)[0]
        remaining_edges_keep = np.delete(grapht[0][2], edges_to_remove, axis=0)
        """

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

        magik_dataset[LABEL_COLUMN_NAME+"_predicted"] = 0

        for index, a_set in enumerate(cluster_sets):
            for value in a_set:
                magik_dataset.loc[original_index[value], LABEL_COLUMN_NAME+"_predicted"] = index + 1
        
        #magik_dataset[LABEL_COLUMN_NAME+"_predicted"] = magik_dataset[LABEL_COLUMN_NAME+"_predicted"].astype(int)

        if LABEL_COLUMN_NAME in magik_dataset.columns:
            magik_dataset[LABEL_COLUMN_NAME] = magik_dataset[LABEL_COLUMN_NAME].astype(int)

        return magik_dataset

    def build_graph(self, full_nodes_dataset, verbose=True, for_predict=False):
        edges_dataframe = pd.DataFrame({
            "distance": [],
            "index_1": [],
            "index_2": [],
            "set": [],
            "same_cluster": []
        })

        #We iterate on all the datasets and we extract all the edges.
        sets = np.unique(full_nodes_dataset[DATASET_COLUMN_NAME])

        if verbose:
            iterator = tqdm.tqdm(sets)
        else:
            iterator = sets

        clusters_extracted_from_dbscan = pd.DataFrame({})

        global_sub_cluster_id = 0

        dbscan_result_by_subcluster_id = {}

        for setid in iterator:
            df_set = full_nodes_dataset[full_nodes_dataset[DATASET_COLUMN_NAME] == setid].copy().reset_index(drop=True)
            eps_values = np.linspace(0, 1, 100)

            results = {}
            dbscan_results = {}

            for eps_value in eps_values:
                if eps_value != 0:
                    dbscan_result = DBSCAN(eps=eps_value, min_samples=1).fit_predict(df_set[['position-x', 'position-y']])

                    if len(set(dbscan_result)) == 1:
                        break

                    if -1 not in dbscan_result:
                        results[eps_value] = metrics.silhouette_score(df_set[['position-x', 'position-y']], dbscan_result)
                        dbscan_results[eps_value] = dbscan_result

            selected_eps_value = max(results, key=results.get)
            dbscan_result = dbscan_results[selected_eps_value]

            for sub_cluster_id in set(dbscan_result):
                # remove excess frames
                localizations_in_subcluster = np.where(dbscan_result == sub_cluster_id)
                cluster_df = df_set.loc[localizations_in_subcluster].copy()
                count = Counter(cluster_df['solution'])
                if for_predict or (len(cluster_df['solution'].unique()) > 1 and (np.array([count[value] for value in count]) > 10).all()):
                    cluster_df[DATASET_COLUMN_NAME] = global_sub_cluster_id
                    dbscan_result_by_subcluster_id[global_sub_cluster_id] = selected_eps_value
                    global_sub_cluster_id += 1
                    clusters_extracted_from_dbscan = clusters_extracted_from_dbscan.append(cluster_df, ignore_index=(not for_predict))

        original_index = clusters_extracted_from_dbscan.index.to_list()
        clusters_extracted_from_dbscan = clusters_extracted_from_dbscan.reset_index(drop=True)
        clusters_extracted_from_dbscan['index'] = clusters_extracted_from_dbscan.index

        #We iterate on all the datasets and we extract all the edges.
        sets = np.unique(clusters_extracted_from_dbscan[DATASET_COLUMN_NAME])

        if verbose:
            iterator = tqdm.tqdm(sets)
        else:
            iterator = sets
        
        for setid in iterator:
            new_edges_dataframe = pd.DataFrame({'index_1': [], 'index_2': [],'distance': [], 'same_cluster': []})
            df_window = clusters_extracted_from_dbscan[clusters_extracted_from_dbscan['set'] == setid].copy().reset_index(drop=True)
            simplices = Delaunay(df_window[[f"{POSITION_COLUMN_NAME}-x", f"{POSITION_COLUMN_NAME}-y", "t"]].values).simplices

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
                f"{POSITION_COLUMN_NAME}-x_x": [],
                f"{POSITION_COLUMN_NAME}-x_y": [],
                f"{POSITION_COLUMN_NAME}-y_x": [],
                f"{POSITION_COLUMN_NAME}-y_y": [],
                'index_x': [],
                'index_y': [],
                LABEL_COLUMN_NAME+"_x": [],
                LABEL_COLUMN_NAME+"_y": [],
                TIME_COLUMN_NAME+"_x": [],
                TIME_COLUMN_NAME+"_y": []
            })

            for edge in list_of_edges:
                x_index = df_window["index"] == edge[0]
                y_index = df_window["index"] == edge[1]

                simplified_cross = simplified_cross.append(pd.DataFrame({
                    f"{POSITION_COLUMN_NAME}-x_x": [df_window[x_index][f"{POSITION_COLUMN_NAME}-x"].values[0]],
                    f"{POSITION_COLUMN_NAME}-x_y": [df_window[y_index][f"{POSITION_COLUMN_NAME}-x"].values[0]],
                    f"{POSITION_COLUMN_NAME}-y_x": [df_window[x_index][f"{POSITION_COLUMN_NAME}-y"].values[0]],
                    f"{POSITION_COLUMN_NAME}-y_y": [df_window[y_index][f"{POSITION_COLUMN_NAME}-y"].values[0]],
                    'index_x': [edge[0]],
                    'index_y': [edge[1]],
                    LABEL_COLUMN_NAME+"_x": [df_window[x_index][f"{LABEL_COLUMN_NAME}"].values[0]],
                    LABEL_COLUMN_NAME+"_y": [df_window[y_index][f"{LABEL_COLUMN_NAME}"].values[0]],
                    TIME_COLUMN_NAME+"_x": [df_window[x_index][f"{TIME_COLUMN_NAME}"].values[0]],
                    TIME_COLUMN_NAME+"_y": [df_window[y_index][f"{TIME_COLUMN_NAME}"].values[0]],
                }), ignore_index=True)

            df_window = simplified_cross.copy()

            """
            def filter_fn(row):
                return [row['index_x'], row['index_y']] in list_of_edges

            df_window = df_window.merge(df_window, how='cross')
            result = df_window.apply(filter_fn, axis=1, )
            df_window = df_window[result]
            """
            df_window = df_window[df_window['index_x'] != df_window['index_y']]
            df_window['distance-x'] = df_window[f"{POSITION_COLUMN_NAME}-x_x"] - df_window[f"{POSITION_COLUMN_NAME}-x_y"]
            df_window['distance-y'] = df_window[f"{POSITION_COLUMN_NAME}-y_x"] - df_window[f"{POSITION_COLUMN_NAME}-y_y"]
            df_window['distance'] = ((df_window['distance-x']**2) + (df_window['distance-y']**2))**(1/2)
            df_window['same_cluster'] = (df_window[LABEL_COLUMN_NAME+"_x"] == df_window[LABEL_COLUMN_NAME+"_y"])
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

        edgefeatures = edges_dataframe[["distance"]].to_numpy()
        sparseadjmtx = edges_dataframe[["index_1", "index_2"]].to_numpy().astype(int)
        nodefeatures = clusters_extracted_from_dbscan[[f"{POSITION_COLUMN_NAME}-x", f"{POSITION_COLUMN_NAME}-y", TIME_COLUMN_NAME]].to_numpy()

        edgeweights = np.ones(sparseadjmtx.shape[0])
        edgeweights = np.stack((np.arange(0, edgeweights.shape[0]), edgeweights), axis=1)

        nfsolution = np.zeros((len(nodefeatures), 1))
        efsolution = edges_dataframe[['same_cluster']].to_numpy().astype(int)

        nodesets = clusters_extracted_from_dbscan[[DATASET_COLUMN_NAME]].to_numpy().astype(int)
        edgesets = edges_dataframe[[DATASET_COLUMN_NAME]].to_numpy().astype(int)
        framesets = clusters_extracted_from_dbscan[[FRAME_COLUMN_NAME]].to_numpy().astype(int)

        global_property = np.zeros(np.unique(clusters_extracted_from_dbscan[DATASET_COLUMN_NAME]).shape[0])

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

    def fit_with_datasets_from_path(self, path):
        """
        !!!
        THIS FILE PERSISTANCE IS TEMPORAL
        !!!
        """

        if os.path.exists('tmp2.tmp2'):
            fileObj = open('tmp2.tmp2', 'rb')
            train_full_graph = pickle.load(fileObj)
            fileObj.close()
        else:
            train_full_graph = self.build_graph(self.get_datasets_from_path(path))
            fileObj = open('tmp2.tmp2', 'wb')
            pickle.dump(train_full_graph, fileObj)
            fileObj.close()

        self.build_network()

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
                graph, labels, _ = data

                min_num_nodes = 500
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
            
                return graph, labels

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

        """
        def OldCustomGetSubGraph():
            def inner(data):
                graph, labels, framesets = data

                framesets = framesets[:,0]
                initial_frame = np.random.choice(np.unique(framesets))
                final_frame = initial_frame + self.hyperparameters["partition_size"]

                if final_frame > np.max(framesets):
                    final_frame = np.max(framesets)
                    initial_frame = final_frame - self.hyperparameters["partition_size"]

                nodeidxs = np.where(np.logical_and(initial_frame <= framesets, framesets < final_frame))

                #node_start = np.random.randint(max(len(graph[0]) - num_nodes, 1))
                #edge_connects_removed_node = np.any(
                #    (graph[2] < node_start) | (graph[2] >= node_start + num_nodes),
                #    axis=-1,
                #)

                edge_connects_removed_node = np.any(~np.isin(graph[2], nodeidxs), axis=-1)

                #node_features = graph[0][node_start : node_start + num_nodes]
                node_features = graph[0][nodeidxs]
                edge_features = graph[1][~edge_connects_removed_node]
                #edge_connections = graph[2][~edge_connects_removed_node] - node_start
                edge_connections = graph[2][~edge_connects_removed_node] - np.min(nodeidxs)
                weights = graph[3][~edge_connects_removed_node]

                #node_labels = labels[0][node_start : node_start + num_nodes]
                node_labels = labels[0][nodeidxs]
                edge_labels = labels[1][~edge_connects_removed_node]
                global_labels = labels[2]

                #print(Counter(np.array(node_labels)))

                return (node_features, edge_features, edge_connections, weights), (
                    node_labels,
                    edge_labels,
                    global_labels,
                )

            return inner
        """

        def CustomGetFeature(full_graph, **kwargs):
            return (
                dt.Value(full_graph)
                >> dt.Lambda(CustomGetSubSet)
                #>> dt.Lambda(CustomGetSubGraph)
                #>> dt.Lambda(CustomDatasetBalancing)
                >> dt.Lambda(
                    AugmentCentroids,
                    rotate=lambda: np.random.rand() * 2 * np.pi,
                    #translate=lambda: np.random.randn(2) * 0.05,
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
            "min_data_size": 256,
            "max_data_size": 257,
            "batch_size": self.hyperparameters["batch_size"],
            "use_multi_inputs": False,
            **magik_variables.properties(),
        }

        generator = ContinuousGraphGenerator(CustomGetFeature(train_full_graph, **magik_variables.properties()), **args)

        with generator:
            self.magik_architecture.fit(generator, epochs=self.hyperparameters["epochs"])

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
