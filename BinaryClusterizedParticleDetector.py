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

logging.disable(logging.WARNING)


class BinaryClusterizedParticleDetector():
    def __init__(self, height=10, width=10):
        self._output_type = "nodes"
        self.magik_architecture = None
        self.hyperparameters = self.__class__.default_hyperparameters()
        self.height = height
        self.width = width

    @classmethod
    def default_hyperparameters(cls):
        return {
            "learning_rate": 0.001,
            "radius": 0.05,
            "nofframes": 7,
            "partition_size": 50
        }

    @classmethod
    def analysis_hyperparameters(cls):
        return {
            "learning_rate": [0.1, 0.01, 0.001],
            "radius": [0.01, 0.025, 0.05, 0.1, 0.25],
            "nofframes": [3, 5, 7, 9, 11],
            "partition_size": [25, 50, 75, 100]
        }

    def build_network(self):
        self.magik_architecture = dt.models.gnns.MAGIK(
            # number of features in each dense encoder layer
            dense_layer_dimensions=(64, 96,),
            # Latent dimension throughout the message passing layers
            base_layer_dimensions=(96, 96, 96),
            number_of_node_features=2,              # Number of node features in the graphs
            number_of_edge_features=1,              # Number of edge features in the graphs
            number_of_node_outputs=1,               # Number of predicted features
            # Activation function for the output layer
            node_output_activation="sigmoid",
            # Output type. Either "edges", "nodes", or "graph"
            output_type=self._output_type,
        )

        self.magik_architecture.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.hyperparameters['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )

        self.magik_architecture.summary()

    def transform_magik_dataframe_to_smlm_dataset(self, magik_dataframe):
        # normalize centroids between 0 and 1
        magik_dataframe.loc[:, magik_dataframe.columns.str.contains(POSITION_COLUMN_NAME)] = (magik_dataframe.loc[:, magik_dataframe.columns.str.contains(POSITION_COLUMN_NAME)] * np.array([self.width, self.height]))

        magik_dataframe = magik_dataframe.rename(columns={
            f"{POSITION_COLUMN_NAME}-x": X_POSITION_COLUMN_NAME,
            f"{POSITION_COLUMN_NAME}-y": Y_POSITION_COLUMN_NAME,
            LABEL_COLUMN_NAME: CLUSTERIZED_COLUMN_NAME,
            LABEL_COLUMN_NAME+"_predicted": CLUSTERIZED_COLUMN_NAME+"_predicted",
        })

        magik_dataframe = magik_dataframe.drop(DATASET_COLUMN_NAME, axis=1)
        return magik_dataframe

    def transform_smlm_dataset_to_magik_dataframe(self, smlm_dataframe, set_number=0):
        smlm_dataframe = smlm_dataframe.rename(columns={
            X_POSITION_COLUMN_NAME: f"{POSITION_COLUMN_NAME}-x",
            Y_POSITION_COLUMN_NAME: f"{POSITION_COLUMN_NAME}-y",
            CLUSTERIZED_COLUMN_NAME: LABEL_COLUMN_NAME,
        })

        if CLUSTER_ID_COLUMN_NAME in smlm_dataframe.columns:
            smlm_dataframe = smlm_dataframe.drop(CLUSTER_ID_COLUMN_NAME, axis=1)
        if PARTICLE_ID_COLUMN_NAME in smlm_dataframe.columns:
            smlm_dataframe = smlm_dataframe.drop(PARTICLE_ID_COLUMN_NAME, axis=1)
        if "Unnamed: 0" in smlm_dataframe.columns:
            smlm_dataframe = smlm_dataframe.drop("Unnamed: 0", axis=1)

        # normalize centroids between 0 and 1
        smlm_dataframe.loc[:, smlm_dataframe.columns.str.contains(POSITION_COLUMN_NAME)] = (smlm_dataframe.loc[:, smlm_dataframe.columns.str.contains(POSITION_COLUMN_NAME)] / np.array([self.width, self.height]))

        smlm_dataframe[DATASET_COLUMN_NAME] = set_number
        smlm_dataframe[LABEL_COLUMN_NAME] = smlm_dataframe[LABEL_COLUMN_NAME].astype(float)

        return smlm_dataframe.reset_index(drop=True)

    def get_dataset_from_path(self, path, set_number=0):
        return self.transform_smlm_dataset_to_magik_dataframe(pd.read_csv(path), set_number=set_number)

    def get_datasets_from_path(self, path):
        file_names = [file_name for file_name in os.listdir(path) if file_name.endswith(".csv")]
        full_dataset = pd.DataFrame({})

        for csv_file_index, csv_file_name in enumerate(file_names):
            full_dataset = full_dataset.append(self.get_dataset_from_path(os.path.join(path, csv_file_name), set_number=csv_file_index))

        return full_dataset.reset_index(drop=True)

    def predict(self, magik_dataset):
        magik_dataset = magik_dataset.copy()

        for frame_index in range(0, max(magik_dataset['frame']), self.hyperparameters["partition_size"]):
            aux_magik_dataset = magik_dataset[magik_dataset['frame'] < frame_index + self.hyperparameters["partition_size"]]
            aux_magik_dataset = aux_magik_dataset[frame_index <= aux_magik_dataset['frame']]
            aux_magik_dataset['frame'] = aux_magik_dataset['frame'] - frame_index
            original_index = aux_magik_dataset.index
            aux_magik_dataset = aux_magik_dataset.copy().reset_index(drop=True)

            grapht = self.build_graph(aux_magik_dataset)

            v = [
                grapht[0][0].reshape(1, grapht[0][0].shape[0], grapht[0][0].shape[1]),
                grapht[0][1].reshape(1, grapht[0][1].shape[0], grapht[0][1].shape[1]),
                grapht[0][2].reshape(1, grapht[0][2].shape[0], grapht[0][2].shape[1]),
                grapht[0][3].reshape(1, grapht[0][3].shape[0], grapht[0][3].shape[1]),
            ]

            magik_dataset.loc[original_index,LABEL_COLUMN_NAME+"_predicted"] = (self.magik_architecture(v).numpy() > 0.5)[0, ...]

        magik_dataset[LABEL_COLUMN_NAME+"_predicted"] = magik_dataset[LABEL_COLUMN_NAME+"_predicted"].astype(int)

        if LABEL_COLUMN_NAME in magik_dataset.columns:
            magik_dataset[LABEL_COLUMN_NAME] = magik_dataset[LABEL_COLUMN_NAME].astype(int)

        return magik_dataset
            #return pred, g, output_node_f, grapht

    def build_graph(self, full_nodes_dataset):
        edges_dataframe = pd.DataFrame({
            "distance": [],
            "index_1": [],
            "index_2": [],
            "set": [],
        })

        full_nodes_dataset = full_nodes_dataset.copy()

        #We iterate on all the datasets and we extract all the edges.
        sets = np.unique(full_nodes_dataset[DATASET_COLUMN_NAME])
        for setid in tqdm.tqdm(sets):
            df_set = full_nodes_dataset[full_nodes_dataset[DATASET_COLUMN_NAME] == setid].copy().reset_index()

            # Create subsets from the frame list, with
            # "nofframes" elements each
            maxframe = range(0, df_set[FRAME_COLUMN_NAME].max() + 1 + self.hyperparameters['nofframes'])

            windows = mit.windowed(maxframe, n=self.hyperparameters['nofframes'], step=1)
            windows = map(
                lambda x: list(filter(partial(is_not, None), x)), windows
            )
            windows = list(windows)[:-2]

            new_edges_dataframe = pd.DataFrame({'index_1': [], 'index_2': [],'distance': []})

            for window in windows:
                # remove excess frames
                window = [elem for elem in window if elem <= df_set[FRAME_COLUMN_NAME].max()]

                df_window = df_set[df_set[FRAME_COLUMN_NAME].isin(window)].copy()
                df_window = df_window.merge(df_window, how='cross')
                df_window = df_window[df_window['index_x'] != df_window['index_y']]
                df_window['distance-x'] = df_window[f"{POSITION_COLUMN_NAME}-x_x"] - df_window[f"{POSITION_COLUMN_NAME}-x_y"]
                df_window['distance-y'] = df_window[f"{POSITION_COLUMN_NAME}-y_x"] - df_window[f"{POSITION_COLUMN_NAME}-y_y"]
                df_window['distance'] = ((df_window['distance-x']**2) + (df_window['distance-y']**2))**(1/2)
                df_window = df_window[df_window['distance'] < self.hyperparameters['radius']]

                edges = [sorted(edge) for edge in df_window[["index_x", "index_y"]].values.tolist()]

                new_edges_dataframe = new_edges_dataframe.append(pd.DataFrame({
                    'index_1': [edge[0] for edge in edges],
                    'index_2': [edge[1] for edge in edges],
                    'distance': [value[0] for value in df_window[["distance"]].values.tolist()]
                }), ignore_index=True)

            new_edges_dataframe = new_edges_dataframe.drop_duplicates()
            new_edges_dataframe['set'] = setid
            edges_dataframe = edges_dataframe.append(new_edges_dataframe, ignore_index=True)

        edgefeatures = edges_dataframe[["distance"]].to_numpy()
        sparseadjmtx = edges_dataframe[["index_1", "index_2"]].to_numpy().astype(int)
        nodefeatures = full_nodes_dataset[[f"{POSITION_COLUMN_NAME}-x", f"{POSITION_COLUMN_NAME}-y"]].to_numpy()

        edgeweights = np.ones(sparseadjmtx.shape[0])
        edgeweights = np.stack((np.arange(0, edgeweights.shape[0]), edgeweights), axis=1)

        nfsolution = full_nodes_dataset[[LABEL_COLUMN_NAME]].to_numpy()
        efsolution = np.zeros((len(edgefeatures), 1))

        nodesets = full_nodes_dataset[[DATASET_COLUMN_NAME]].to_numpy().astype(int)
        edgesets = edges_dataframe[[DATASET_COLUMN_NAME]].to_numpy().astype(int)
        framesets = full_nodes_dataset[[FRAME_COLUMN_NAME]].to_numpy().astype(int)

        global_property = np.zeros(np.unique(full_nodes_dataset[DATASET_COLUMN_NAME]).shape[0])

        return (
            (nodefeatures, edgefeatures, sparseadjmtx, edgeweights),
            (nfsolution, efsolution, global_property),
            (nodesets, edgesets, framesets)
        )

    def fit_with_datasets_from_path(self, path):
        self.build_network()
        train_full_graph = self.build_graph(self.get_datasets_from_path(os.path.join(path, 'train')))

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
                graph, labels, framesets = data

                framesets = framesets[:,0]
                initial_frame = np.random.choice(np.unique(framesets))
                final_frame = initial_frame + self.hyperparameters["partition_size"]

                if final_frame > np.max(framesets):
                    final_frame = np.max(framesets)
                    initial_frame = final_frame - self.hyperparameters["partition_size"]

                nodeidxs = np.where(np.logical_and(initial_frame <= framesets, framesets < final_frame))

                """
                node_start = np.random.randint(max(len(graph[0]) - num_nodes, 1))

                edge_connects_removed_node = np.any(
                    (graph[2] < node_start) | (graph[2] >= node_start + num_nodes),
                    axis=-1,
                )
                """

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

                return (node_features, edge_features, edge_connections, weights), (
                    node_labels,
                    edge_labels,
                    global_labels,
                )

            return inner

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
                    flip_y=lambda: np.random.randint(2),
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
            "min_data_size": 127,
            "max_data_size": 128,
            "batch_size": 1,
            "use_multi_inputs": False,
            **magik_variables.properties(),
        }

        generator = ContinuousGraphGenerator(CustomGetFeature(train_full_graph, **magik_variables.properties()), **args)

        with generator:
            self.magik_architecture.fit(generator, epochs=10)
