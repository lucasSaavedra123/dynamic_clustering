from CONSTANTS import *
from deeptrack.models.gnns.augmentations import AugmentCentroids, NodeDropout
from deeptrack.models.gnns.generators import GraphExtractor, ContinuousGraphGenerator
import deeptrack as dt
import tensorflow as tf
import numpy as np
import pandas as pd
import os
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
logging.disable(logging.WARNING)


class SubClusterLinking():
    def __init__(self, binary_clusterized_particle_detector, height=10, width=10):
        self._output_type = "edges"
        self.magik_architecture = None
        self.hyperparameters = self.__class__.default_hyperparameters()
        self.height = height
        self.width = width
        self.binary_clusterized_particle_detector = binary_clusterized_particle_detector

    @classmethod
    def default_hyperparameters(cls):
        return {
            "learning_rate": 0.001,
            "radius": 0.25,
            "nofframes": 25,
            "partition_size": 100,
            "epochs": 25,
            "batch_size": 4,
        }

    @classmethod
    def analysis_hyperparameters(cls):
        return {
            "learning_rate": [0.1, 0.01, 0.001],
            "radius": [0.01, 0.025, 0.05, 0.1, 0.25],
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
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            #loss="mse",
            metrics=['accuracy', tf.keras.metrics.AUC()]
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

        for csv_file_name in file_names[0:20]:
            set_dataframe = self.get_dataset_from_path(os.path.join(path, csv_file_name), set_number=set_index)

            if not set_dataframe.empty:
                full_dataset = full_dataset.append(set_dataframe)
                set_index += 1

        return full_dataset.reset_index(drop=True)

    def predict(self, magik_dataset, threshold=0.5):
        magik_dataset = magik_dataset.copy()

        grapht, localization_to_subcluster_id = self.build_graph(magik_dataset, return_localization_centroid_to_dictionary=True)

        v = [
            grapht[0][0].reshape(1, grapht[0][0].shape[0], grapht[0][0].shape[1]),
            grapht[0][1].reshape(1, grapht[0][1].shape[0], grapht[0][1].shape[1]),
            grapht[0][2].reshape(1, grapht[0][2].shape[0], grapht[0][2].shape[1]),
            grapht[0][3].reshape(1, grapht[0][3].shape[0], grapht[0][3].shape[1]),
        ]

        predictions = (self.magik_architecture(v).numpy() > threshold)[0, ...]
        edges_to_remove = np.where(predictions == 0)[0]
        remaining_edges_keep = np.delete(grapht[0][2], edges_to_remove, axis=0)

        magik_dataset.loc[list(localization_to_subcluster_id.keys()), LABEL_COLUMN_NAME+"_predicted"] = list(localization_to_subcluster_id.values())

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

        mapping_sub_cluster_to_cluster = {}

        for index, a_set in enumerate(cluster_sets):
            for value in a_set:
                mapping_sub_cluster_to_cluster[value] = index

        magik_dataset[LABEL_COLUMN_NAME+"_predicted"] = magik_dataset[LABEL_COLUMN_NAME+"_predicted"].map(mapping_sub_cluster_to_cluster)
        #magik_dataset[LABEL_COLUMN_NAME+"_predicted"] = magik_dataset[LABEL_COLUMN_NAME+"_predicted"].astype(int)

        if LABEL_COLUMN_NAME in magik_dataset.columns:
            magik_dataset[LABEL_COLUMN_NAME] = magik_dataset[LABEL_COLUMN_NAME].astype(int)

        return magik_dataset

    def build_graph(self, full_nodes_dataset, verbose=True, return_localization_centroid_to_dictionary=False):
        edges_dataframe = pd.DataFrame({
            "distance": [],
            "index_1": [],
            "index_2": [],
            "set": [],
            "same_cluster": []
        })

        number_of_centroids = 0

        full_nodes_dataset = full_nodes_dataset.copy()
        full_centroids_dataset = pd.DataFrame({})

        #We iterate on all the datasets and we extract all the edges.
        sets = np.unique(full_nodes_dataset[DATASET_COLUMN_NAME])

        if verbose:
            iterator = tqdm.tqdm(sets)
        else:
            iterator = sets

        for setid in iterator:
            #df_set = full_nodes_dataset[full_nodes_dataset[DATASET_COLUMN_NAME] == setid].copy()
            df_set = full_nodes_dataset[full_nodes_dataset[DATASET_COLUMN_NAME] == setid].copy().reset_index()

            """
            new_graph = self.binary_clusterized_particle_detector.build_graph(df_set, verbose=False)

            edges_adjacency = new_graph[0][2]

            cluster_sets = []

            for i in range(len(edges_adjacency)):
                cluster_assigned = False
                if len(cluster_sets) == 0:
                    cluster_sets.append(set([edges_adjacency[i][0], edges_adjacency[i][1]]))
                else:
                    for index, s in enumerate(cluster_sets):
                        if edges_adjacency[i][0] in s or edges_adjacency[i][1] in s:
                            s.add(edges_adjacency[i][0])
                            s.add(edges_adjacency[i][1])
                            cluster_assigned = True
                            break

                    if not cluster_assigned:
                        cluster_sets.append(set([edges_adjacency[i][0], edges_adjacency[i][1]]))

            df_set['sub_cluster_id'] = 0

            for index, a_set in enumerate(cluster_sets):
                for value in a_set:
                    df_set.loc[value, 'sub_cluster_id'] = index + 1

            localization_to_centroid_dictionary = {}
            localization_to_centroid_dictionary.update({index: df_set.loc[index, 'sub_cluster_id'] - min(df_set.index) for index in df_set.index})

            df_set = df_set.groupby('sub_cluster_id', as_index=False).agg({
                f"{POSITION_COLUMN_NAME}-x": 'mean',
                f"{POSITION_COLUMN_NAME}-y": 'mean',
                TIME_COLUMN_NAME: 'mean',
                LABEL_COLUMN_NAME: lambda x: x.value_counts().index[0],
                FRAME_COLUMN_NAME: 'mean'
            })

            df_set = df_set.reset_index(drop=True)
            df_set['frame'] = df_set['frame'].astype(int)
            df_set['index'] = df_set.index + number_of_centroids
            df_set['set'] = setid
            number_of_centroids += len(df_set)

            full_centroids_dataset = full_centroids_dataset.append(df_set, ignore_index=True)

            new_edges_dataframe = pd.DataFrame({'index_1': [], 'index_2': [],'distance': [], 'same_cluster': []})

            new_edges_dataframe['set'] = setid

            df_set = df_set.merge(df_set, how='cross')
            df_set = df_set[df_set['index_x'] != df_set['index_y']]
            df_set['distance-x'] = df_set[f"{POSITION_COLUMN_NAME}-x_x"] - df_set[f"{POSITION_COLUMN_NAME}-x_y"]
            df_set['distance-y'] = df_set[f"{POSITION_COLUMN_NAME}-y_x"] - df_set[f"{POSITION_COLUMN_NAME}-y_y"]
            df_set['distance'] = ((df_set['distance-x']**2) + (df_set['distance-y']**2))**(1/2)
            df_set['same_cluster'] = (df_set[LABEL_COLUMN_NAME+"_x"] == df_set[LABEL_COLUMN_NAME+"_y"])
            #df_set = df_set[df_set['distance'] < self.hyperparameters['radius']]

            edges = [sorted(edge) for edge in df_set[["index_x", "index_y"]].values.tolist()]

            new_edges_dataframe = new_edges_dataframe.append(pd.DataFrame({
                'index_1': [edge[0] for edge in edges],
                'index_2': [edge[1] for edge in edges],
                'distance': [value[0] for value in df_set[["distance"]].values.tolist()],
                'same_cluster': [value[0] for value in df_set[["same_cluster"]].values.tolist()],
            }), ignore_index=True)
            """

            # Create subsets from the frame list, with
            # "nofframes" elements each
            maxframe = range(0, df_set[FRAME_COLUMN_NAME].max() + 1 + self.hyperparameters['nofframes'])

            windows = mit.windowed(maxframe, n=self.hyperparameters['nofframes'], step=1)
            windows = map(
                lambda x: list(filter(partial(is_not, None), x)), windows
            )
            windows = list(windows)[:-2]

            new_edges_dataframe = pd.DataFrame({'index_1': [], 'index_2': [],'distance': [], 'same_cluster': []})

            for window in windows:
                # remove excess frames
                window = [elem for elem in window if elem <= df_set[FRAME_COLUMN_NAME].max()]

                df_window = df_set[df_set[FRAME_COLUMN_NAME].isin(window)].copy()
                df_window = df_window.merge(df_window, how='cross')
                df_window = df_window[df_window['index_x'] != df_window['index_y']]
                df_window['distance-x'] = df_window[f"{POSITION_COLUMN_NAME}-x_x"] - df_window[f"{POSITION_COLUMN_NAME}-x_y"]
                df_window['distance-y'] = df_window[f"{POSITION_COLUMN_NAME}-y_x"] - df_window[f"{POSITION_COLUMN_NAME}-y_y"]
                df_window['distance'] = ((df_window['distance-x']**2) + (df_window['distance-y']**2))**(1/2)
                df_window['same_cluster'] = (df_window[LABEL_COLUMN_NAME+"_x"] == df_window[LABEL_COLUMN_NAME+"_y"])
                df_window = df_window[df_window['distance'] < self.hyperparameters['radius']]

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
        #nodefeatures = full_centroids_dataset[[f"{POSITION_COLUMN_NAME}-x", f"{POSITION_COLUMN_NAME}-y", f"{TIME_COLUMN_NAME}"]].to_numpy()
        nodefeatures = full_nodes_dataset[[f"{POSITION_COLUMN_NAME}-x", f"{POSITION_COLUMN_NAME}-y"]].to_numpy()

        edgeweights = np.ones(sparseadjmtx.shape[0])
        edgeweights = np.stack((np.arange(0, edgeweights.shape[0]), edgeweights), axis=1)

        nfsolution = np.zeros((len(nodefeatures), 1))
        efsolution = edges_dataframe[['same_cluster']].to_numpy().astype(int)

        #nodesets = full_centroids_dataset[[DATASET_COLUMN_NAME]].to_numpy().astype(int)
        nodesets = full_nodes_dataset[[DATASET_COLUMN_NAME]].to_numpy().astype(int)
        edgesets = edges_dataframe[[DATASET_COLUMN_NAME]].to_numpy().astype(int)
        #framesets = full_centroids_dataset[[FRAME_COLUMN_NAME]].to_numpy().astype(int)
        framesets = full_nodes_dataset[[FRAME_COLUMN_NAME]].to_numpy().astype(int)

        #global_property = np.zeros(np.unique(full_centroids_dataset[DATASET_COLUMN_NAME]).shape[0])
        global_property = np.zeros(np.unique(full_nodes_dataset[DATASET_COLUMN_NAME]).shape[0])

        if return_localization_centroid_to_dictionary:
            return (
                (nodefeatures, edgefeatures, sparseadjmtx, edgeweights),
                (nfsolution, efsolution, global_property),
                (nodesets, edgesets, framesets)
            ), localization_to_centroid_dictionary
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

        def CustomGetSubSet(randset):
            """
            Returns a function that takes a graph and returns a
            random subset of the graph.
            """

            def inner(data):
                graph, labels, sets = data

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

                return (node_features, edge_features, edge_connections, weights), (
                    node_labels,
                    edge_labels,
                    glob_labels,
                ), (frame_sets) #, (node_sets, edge_sets, frame_sets)

            return inner

        def CustomGetSubGraph():
            def inner(data):
                graph, labels, _ = data

                min_num_nodes = 350
                max_num_nodes = 1000

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

                    #print(Counter(np.array(labels[1])[:,0]))

                    retry = len(np.array(edge_labels)) == 0

                return (node_features, edge_features, edge_connections, weights), (
                    node_labels,
                    edge_labels,
                    global_labels,
                )

            return inner

        def CustomDatasetBalancing():
            def inner(data):
                graph, labels = data

                number_of_clusterized_nodes = np.sum(np.array((labels[0][:, 0] == 1)) * 1)
                number_of_non_clusterized_nodes = np.sum(np.array(labels[0][:, 0] == 0) * 1)

                if number_of_clusterized_nodes > number_of_non_clusterized_nodes:
                    nodeidxs = np.array(np.where(labels[0][:, 0] == 1)[0])

                    number_of_nodes_to_select = number_of_non_clusterized_nodes

                    nodes_to_select = np.random.choice(nodeidxs, size=number_of_nodes_to_select, replace=False)
                    nodes_to_select = np.append(nodes_to_select, np.array(np.where(labels[0][:, 0] == 0)[0]))

                    id_to_new_id = {}

                    for index, value in enumerate(nodes_to_select):
                        id_to_new_id[value] = index

                    edge_connects_removed_node = np.any( ~np.isin(graph[2], nodes_to_select), axis=-1)

                    node_features = graph[0][nodes_to_select]
                    edge_features = graph[1][~edge_connects_removed_node]
                    edge_connections = graph[2][~edge_connects_removed_node]

                    edge_connections = np.vectorize(id_to_new_id.get)(edge_connections)

                    weights = graph[3][~edge_connects_removed_node]

                    node_labels = labels[0][nodes_to_select]
                    edge_labels = labels[1][~edge_connects_removed_node]
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
                >> dt.Lambda(CustomGetSubSet, randset=lambda: np.random.randint(np.max(full_graph[-1][0][:, 0]) + 1),)
                >> dt.Lambda(CustomGetSubGraph)
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
