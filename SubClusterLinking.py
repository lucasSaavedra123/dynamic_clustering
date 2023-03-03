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
from BinaryClusterizedParticleDetector import BinaryClusterizedParticleDetector

logging.disable(logging.WARNING)


class SubClusterLinking():
    def __init__(self, binary_classificator=None, height=10, width=10, radius=0.2, nofframes=10):
        self._output_type = "edges"

        self.magik_variables = dt.DummyFeature(
            radius=radius,
            output_type=self._output_type,
            nofframes=nofframes,  # time window to associate nodes (in frames)
        )

        self.nofframes = nofframes
        self.radius = radius
        self.height = height
        self.width = width

        if binary_classificator is None:
            self.binary_classificator = BinaryClusterizedParticleDetector(height, width, radius, nofframes)
        else:
            self.binary_classificator = binary_classificator

        self.magik_architecture = None

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
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
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
            CLUSTER_ID_COLUMN_NAME: LABEL_COLUMN_NAME,
        })

        smlm_dataframe = smlm_dataframe[smlm_dataframe[CLUSTERIZED_COLUMN_NAME] == 1]
        smlm_dataframe = smlm_dataframe.drop(CLUSTERIZED_COLUMN_NAME, axis=1)

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

    def predict(self, magik_dataset, number_of_frames_per_step=None):
        magik_dataset = magik_dataset.copy()

        if number_of_frames_per_step is None:
            number_of_frames_per_step = max(magik_dataset['frame'])

        for frame_index in range(0, max(magik_dataset['frame']), number_of_frames_per_step):
            aux_magik_dataset = magik_dataset[magik_dataset['frame'] < frame_index + number_of_frames_per_step]
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
            LABEL_COLUMN_NAME: []
        })

        full_nodes_dataset = full_nodes_dataset.copy()

        #We iterate on all the datasets and we extract all the edges.
        sets = np.unique(full_nodes_dataset[DATASET_COLUMN_NAME])
        for setid in tqdm.tqdm(sets):
            df_set = full_nodes_dataset[full_nodes_dataset[DATASET_COLUMN_NAME] == setid].copy().reset_index()
    
            new_graph = self.binary_classificator.build_graph(df_set)[0]

            edges_features = new_graph[1][0]
            edges_adjacency = new_graph[2][0]

            sets = []

            for i in range(len(edges_adjacency)):
                cluster_assigned = False
                if len(sets) == 0:
                    sets.append(set([edges_adjacency[i][0], edges_adjacency[i][1]]))
                else:
                    for index, s in enumerate(sets):
                        if edges_adjacency[i][0] in s or edges_adjacency[i][1] in s:
                            s.add(edges_adjacency[i][0])
                            s.add(edges_adjacency[i][1])
                            cluster_assigned = True
                            break

                    if not cluster_assigned:
                        sets.append(set([edges_adjacency[i][0], edges_adjacency[i][1]]))

            df_set['sub_cluster_id'] = 0

            for index, a_set in enumerate(sets):
                for value in a_set:
                    df_set.loc[value, 'sub_cluster_id'] = index + 1

            df_set = df_set.groupby('sub_cluster_id', as_index=False).agg({
                'x': 'mean',
                'y': 'mean',
                't': 'mean',
                'cluster_id': lambda x: x.value_counts().index[0],
                'frame': 'mean'
            })

            df_set['frame'] = df_set['frame'].astype(int)

            # Create subsets from the frame list, with
            # "nofframes" elements each
            maxframe = range(0, df_set[FRAME_COLUMN_NAME].max() + 1 + self.nofframes)

            windows = mit.windowed(maxframe, n=maxframe, step=1)
            windows = map(
                lambda x: list(filter(partial(is_not, None), x)), windows
            )
            windows = list(windows)[:-2]

            new_edges_dataframe = pd.DataFrame({'index_1': [], 'index_2': [],'distance': []})

            for window in windows:
                # remove excess frames
                window = [elem for elem in window if elem <= df_set["frame"].max()]

                df_window = df_set[df_set[FRAME_COLUMN_NAME].isin(window)].copy()
                df_window = df_window.merge(df_window, how='cross')
                df_window = df_window[df_window['index_x'] != df_window['index_y']]
                df_window['distance-x'] = df_window[f"{POSITION_COLUMN_NAME}-x_x"] - df_window[f"{POSITION_COLUMN_NAME}-x_y"]
                df_window['distance-y'] = df_window[f"{POSITION_COLUMN_NAME}-y_x"] - df_window[f"{POSITION_COLUMN_NAME}-y_y"]
                df_window['distance'] = ((df_window['distance-x']**2) + (df_window['distance-y']**2))**(1/2)
                df_window = df_window[df_window['distance'] < self.radius]

                edges = [sorted(edge) for edge in df_window[["index_x", "index_y"]].values.tolist()]

                new_edges_dataframe = new_edges_dataframe.append(pd.DataFrame({
                    'index_1': [edge[0] for edge in edges],
                    'index_2': [edge[1] for edge in edges],
                    'distance': [value[0] for value in df_window[["distance"]].values.tolist()],
                    LABEL_COLUMN_NAME: [value[0] for value in (df_window["cluster_id_x"] == df_window["cluster_id_y"]).astype(int).values]
                }))

            new_edges_dataframe = new_edges_dataframe.drop_duplicates()
            new_edges_dataframe['set'] = setid
            edges_dataframe = edges_dataframe.append(new_edges_dataframe)

        edgefeatures = edges_dataframe[["distance"]].to_numpy()
        sparseadjmtx = edges_dataframe[["index_1", "index_2"]].to_numpy().astype(int)
        nodefeatures = full_nodes_dataset[[f"{POSITION_COLUMN_NAME}-x", f"{POSITION_COLUMN_NAME}-y"]].to_numpy()

        edgeweights = np.ones(sparseadjmtx.shape[0])
        edgeweights = np.stack((np.arange(0, edgeweights.shape[0]), edgeweights), axis=1)

        nfsolution = np.zeros((len(edgefeatures), 1))
        efsolution = edges_dataframe[[LABEL_COLUMN_NAME]].to_numpy()

        nodesets = full_nodes_dataset[[DATASET_COLUMN_NAME]].to_numpy().astype(int)
        edgesets = edges_dataframe[[DATASET_COLUMN_NAME]].to_numpy().astype(int)

        global_property = np.zeros(np.unique(full_nodes_dataset[DATASET_COLUMN_NAME]).shape[0])
            
        return (
            (nodefeatures, edgefeatures, sparseadjmtx, edgeweights),
            (nfsolution, efsolution, global_property),
            (nodesets, edgesets)
        )

    def fit_with_datasets_from_path(self, path):
        full_nodes_dataset = self.get_datasets_from_path(path)
        self.build_network()

        a_full_graph = self.build_graph(full_nodes_dataset)

        def CustomGetSubGraph(num_nodes):
            def inner(data):
                graph, labels = data

                node_start = np.random.randint(max(len(graph[0]) - num_nodes, 1))

                edge_connects_removed_node = np.any(
                    (graph[2] < node_start) | (graph[2] >= node_start + num_nodes),
                    axis=-1,
                )

                node_features = graph[0][node_start : node_start + num_nodes]
                edge_features = graph[1][~edge_connects_removed_node]
                edge_connections = graph[2][~edge_connects_removed_node] - node_start
                weights = graph[3][~edge_connects_removed_node]

                node_labels = labels[0][node_start : node_start + num_nodes]
                edge_labels = labels[1][~edge_connects_removed_node]
                global_labels = labels[2]

                return (node_features, edge_features, edge_connections, weights), (
                    node_labels,
                    edge_labels,
                    global_labels,
                )

            return inner

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

                return (node_features, edge_features, edge_connections, weights), (
                    node_labels,
                    edge_labels,
                    glob_labels,
                )#, (node_sets, edge_sets)

            return inner

        def CustomGetFeature(full_graph, **kwargs):
            return (
                dt.Value(full_graph)
                >> dt.Lambda(
                    CustomGetSubSet,
                    randset=lambda: np.random.randint(np.max(full_graph[-1][0][:, 0]) + 1),
                )
                >> dt.Lambda(
                    CustomGetSubGraph,
                    num_nodes=lambda min_num_nodes, max_num_nodes: np.random.randint(
                        min_num_nodes, max_num_nodes
                    ),
                    #node_start=lambda num_nodes: np.random.randint(max(len(full_graph[0][0]) - num_nodes, 1)),
                    min_num_nodes=700,
                    max_num_nodes=1500,
                )
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


        feature = CustomGetFeature(a_full_graph, **self.magik_variables.properties())

        args = {
            "batch_function": lambda graph: graph[0],
            "label_function": lambda graph: graph[1],
            "min_data_size": 512,
            "max_data_size": 513,
            "batch_size": 8,
            "use_multi_inputs": False,
            **self.magik_variables.properties(),
        }

        generator = ContinuousGraphGenerator(feature, **args)

        with generator:
            self.magik_architecture.fit(generator, epochs=10)
