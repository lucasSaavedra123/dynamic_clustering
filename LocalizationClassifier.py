from collections import Counter
import os
import more_itertools as mit
import tqdm
from operator import is_not
from functools import partial
import pickle

from deeptrack.models.gnns.augmentations import NodeDropout
from deeptrack.models.gnns.generators import ContinuousGraphGenerator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import deeptrack as dt
import tensorflow as tf
import numpy as np
import pandas as pd
import ghostml

from CONSTANTS import *


class LocalizationClassifier():
    @classmethod
    def default_hyperparameters(cls):
        return {
            "learning_rate": 0.001,
            "radius": 0.05,
            "nofframes": 11,
            "partition_size": 100,
            "epochs": 5,
            "batch_size": 4,
        }

    @classmethod
    def analysis_hyperparameters(cls):
        return {
            #"learning_rate": [0.01, 0.001, 0.001],
            "radius": [0.05, 0.1, 0.25],
            "nofframes": [11,13,15,17,19,21],
            "batch_size": [1,2,4]
        }

    def __init__(self, height=10, width=10):
        self._output_type = "nodes"

        self.magik_architecture = None
        self.threshold = 0.5

        self.hyperparameters = self.__class__.default_hyperparameters()
        self.height = height
        self.width = width

    @property
    def node_features(self):
        return [MAGIK_X_POSITION_COLUMN_NAME, MAGIK_Y_POSITION_COLUMN_NAME]

    @property
    def edge_features(self):
        return ["distance"]

    def build_network(self):
        self.magik_architecture = dt.models.gnns.MAGIK(
            dense_layer_dimensions=(64, 96,),
            base_layer_dimensions=(96, 96, 96),
            number_of_node_features=len(self.node_features),
            number_of_edge_features=len(self.edge_features),
            number_of_node_outputs=1,
            node_output_activation="sigmoid",
            output_type=self._output_type,
        )

        self.magik_architecture.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.hyperparameters['learning_rate']),
            #loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            loss="mse",
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )

        self.magik_architecture.summary()

    def transform_smlm_dataset_to_magik_dataframe(self, smlm_dataframe, set_number=0):
        smlm_dataframe = smlm_dataframe.rename(columns={
            X_POSITION_COLUMN_NAME: MAGIK_X_POSITION_COLUMN_NAME,
            Y_POSITION_COLUMN_NAME: MAGIK_Y_POSITION_COLUMN_NAME,
            CLUSTERIZED_COLUMN_NAME: MAGIK_LABEL_COLUMN_NAME,
            CLUSTERIZED_COLUMN_NAME+"_predicted": MAGIK_LABEL_COLUMN_NAME_PREDICTED
        })

        smlm_dataframe = smlm_dataframe.drop([CLUSTER_ID_COLUMN_NAME, PARTICLE_ID_COLUMN_NAME, "Unnamed: 0"], axis=1, errors="ignore")
        smlm_dataframe.loc[:, smlm_dataframe.columns.str.contains(MAGIK_POSITION_COLUMN_NAME)] = (smlm_dataframe.loc[:, smlm_dataframe.columns.str.contains(MAGIK_POSITION_COLUMN_NAME)] / np.array([self.width, self.height]))

        smlm_dataframe[MAGIK_DATASET_COLUMN_NAME] = set_number
        smlm_dataframe[MAGIK_LABEL_COLUMN_NAME] = smlm_dataframe[MAGIK_LABEL_COLUMN_NAME].astype(int)

        return smlm_dataframe.reset_index(drop=True)

    def transform_magik_dataframe_to_smlm_dataset(self, magik_dataframe):
        magik_dataframe.loc[:, magik_dataframe.columns.str.contains(MAGIK_POSITION_COLUMN_NAME)] = (magik_dataframe.loc[:, magik_dataframe.columns.str.contains(MAGIK_POSITION_COLUMN_NAME)] * np.array([self.width, self.height]))

        magik_dataframe = magik_dataframe.rename(columns={
            MAGIK_X_POSITION_COLUMN_NAME: X_POSITION_COLUMN_NAME,
            MAGIK_Y_POSITION_COLUMN_NAME: Y_POSITION_COLUMN_NAME,
            MAGIK_LABEL_COLUMN_NAME: CLUSTERIZED_COLUMN_NAME,
            MAGIK_LABEL_COLUMN_NAME_PREDICTED: CLUSTERIZED_COLUMN_NAME+"_predicted",
        })

        magik_dataframe = magik_dataframe.drop(MAGIK_DATASET_COLUMN_NAME, axis=1)

        return magik_dataframe.reset_index(drop=True)
  
    def get_dataset_from_path(self, path, set_number=0):
        return self.transform_smlm_dataset_to_magik_dataframe(pd.read_csv(path), set_number=set_number)

    def get_dataset_file_paths_from(self, path):
        return [os.path.join(path, file_name) for file_name in os.listdir(path) if file_name.endswith(".csv")]

    def get_datasets_from_path(self, path):
        full_dataset = pd.DataFrame({})

        for csv_file_index, csv_file_path in enumerate(self.get_dataset_file_paths_from(path)):
            full_dataset = full_dataset.append(self.get_dataset_from_path(csv_file_path, set_number=csv_file_index), ignore_index=True)

        return full_dataset.reset_index(drop=True)

    def predict(self, magik_dataset, apply_threshold=True):
        magik_dataset = magik_dataset.copy()

        for frame_index in range(0, max(magik_dataset[FRAME_COLUMN_NAME]), self.hyperparameters["partition_size"]):
            aux_magik_dataset = magik_dataset[magik_dataset[FRAME_COLUMN_NAME] < frame_index + self.hyperparameters["partition_size"]]
            aux_magik_dataset = aux_magik_dataset[frame_index <= aux_magik_dataset[FRAME_COLUMN_NAME]]
            aux_magik_dataset[FRAME_COLUMN_NAME] = aux_magik_dataset[FRAME_COLUMN_NAME] - frame_index
            original_index = aux_magik_dataset.index
            aux_magik_dataset = aux_magik_dataset.copy().reset_index(drop=True)

            grapht = self.build_graph(aux_magik_dataset)

            v = [
                grapht[0][0].reshape(1, grapht[0][0].shape[0], grapht[0][0].shape[1]),
                grapht[0][1].reshape(1, grapht[0][1].shape[0], grapht[0][1].shape[1]),
                grapht[0][2].reshape(1, grapht[0][2].shape[0], grapht[0][2].shape[1]),
                grapht[0][3].reshape(1, grapht[0][3].shape[0], grapht[0][3].shape[1]),
            ]

            if apply_threshold:
                magik_dataset.loc[original_index,MAGIK_LABEL_COLUMN_NAME_PREDICTED] = (self.magik_architecture(v).numpy() > self.threshold)[0, ...]
            else:
                magik_dataset.loc[original_index,MAGIK_LABEL_COLUMN_NAME_PREDICTED] = (self.magik_architecture(v).numpy())[0, ...]

        if apply_threshold:
            magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED] = magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED].astype(int)
        else:
            magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED] = magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED].astype(float)

        return magik_dataset

    def build_graph(self, full_nodes_dataset, verbose=True):
        edges_dataframe = pd.DataFrame({
            "distance": [],
            "index_1": [],
            "index_2": [],
            MAGIK_DATASET_COLUMN_NAME: [],
        })

        full_nodes_dataset = full_nodes_dataset.copy()

        sets = np.unique(full_nodes_dataset[MAGIK_DATASET_COLUMN_NAME])

        if verbose:
            iterator = tqdm.tqdm(sets)
        else:
            iterator = sets

        for setid in iterator:
            df_set = full_nodes_dataset[full_nodes_dataset[MAGIK_DATASET_COLUMN_NAME] == setid].copy().reset_index()

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
                df_window['distance-x'] = df_window[f"{MAGIK_X_POSITION_COLUMN_NAME}_x"] - df_window[f"{MAGIK_X_POSITION_COLUMN_NAME}_y"]
                df_window['distance-y'] = df_window[f"{MAGIK_Y_POSITION_COLUMN_NAME}_x"] - df_window[f"{MAGIK_Y_POSITION_COLUMN_NAME}_y"]
                df_window['distance'] = ((df_window['distance-x']**2) + (df_window['distance-y']**2))**(1/2)
                #df_window['distance'] = np.linalg.norm(df_window.loc[:, [f"{MAGIK_X_POSITION_COLUMN_NAME}_x", f"{MAGIK_Y_POSITION_COLUMN_NAME}_x"]].values - df_window.loc[:, [f"{MAGIK_X_POSITION_COLUMN_NAME}_y", f"{MAGIK_Y_POSITION_COLUMN_NAME}_y"]].values, axis=1)
                df_window = df_window[df_window['distance'] < self.hyperparameters['radius']]

                edges = [sorted(edge) for edge in df_window[["index_x", "index_y"]].values.tolist()]

                new_edges_dataframe = new_edges_dataframe.append(pd.DataFrame({
                    'index_1': [edge[0] for edge in edges],
                    'index_2': [edge[1] for edge in edges],
                    'distance': [value[0] for value in df_window[["distance"]].values.tolist()]
                }), ignore_index=True)

            new_edges_dataframe = new_edges_dataframe.drop_duplicates()
            new_edges_dataframe[MAGIK_DATASET_COLUMN_NAME] = setid
            edges_dataframe = edges_dataframe.append(new_edges_dataframe, ignore_index=True)

        edgefeatures = edges_dataframe[self.edge_features].to_numpy()
        sparseadjmtx = edges_dataframe[["index_1", "index_2"]].to_numpy().astype(int)
        nodefeatures = full_nodes_dataset[self.node_features].to_numpy()

        edgeweights = np.ones(sparseadjmtx.shape[0])
        edgeweights = np.stack((np.arange(0, edgeweights.shape[0]), edgeweights), axis=1)

        nfsolution = full_nodes_dataset[[MAGIK_LABEL_COLUMN_NAME]].to_numpy()
        efsolution = np.zeros((len(edgefeatures), 1))

        nodesets = full_nodes_dataset[[MAGIK_DATASET_COLUMN_NAME]].to_numpy().astype(int)
        edgesets = edges_dataframe[[MAGIK_DATASET_COLUMN_NAME]].to_numpy().astype(int)
        framesets = full_nodes_dataset[[FRAME_COLUMN_NAME]].to_numpy().astype(int)

        global_property = np.zeros(np.unique(full_nodes_dataset[MAGIK_DATASET_COLUMN_NAME]).shape[0])

        return (
            (nodefeatures, edgefeatures, sparseadjmtx, edgeweights),
            (nfsolution, efsolution, global_property),
            (nodesets, edgesets, framesets)
        )
    
    @property
    def train_full_graph_file_name(self):
        return f"node_classifier_radius_{self.hyperparameters['radius']}_nofframes_{self.hyperparameters['nofframes']}.tmp"

    @property
    def model_file_name(self):
        return f"node_classifier_radius_{self.hyperparameters['radius']}_nofframes_{self.hyperparameters['nofframes']}.h5"

    @property
    def threshold_file_name(self):
        return f"node_classifier_radius_{self.hyperparameters['radius']}_nofframes_{self.hyperparameters['nofframes']}_partition_{self.hyperparameters['partition_size']}.bin"

    @property
    def predictions_file_name(self):
        return f"node_classifier_radius_{self.hyperparameters['radius']}_nofframes_{self.hyperparameters['nofframes']}_partition_{self.hyperparameters['partition_size']}.csv"

    def test_with_datasets_from_path(self, path, plot=False, apply_threshold=True, save_result=False, save_predictions=False):
        true = []
        pred = []

        for csv_file_name in self.get_dataset_file_paths_from(path):
            r = self.predict(self.get_dataset_from_path(csv_file_name), apply_threshold=apply_threshold)

            if save_predictions:
                r.to_csv(csv_file_name+f"_predicted_with_{self.hyperparameters['radius']}_nofframes_{self.hyperparameters['nofframes']}_partition_{self.hyperparameters['partition_size']}.csv", index=False)

            true += r[MAGIK_LABEL_COLUMN_NAME].values.tolist()
            pred += r[MAGIK_LABEL_COLUMN_NAME_PREDICTED].values.tolist()

        if save_result:
            pd.DataFrame({
                'true': true,
                'pred': pred
            }).to_csv(self.predictions_file_name, index=False)

        if plot:
            self.plot_confusion_matrix(true, pred)

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

        if not os.path.exists(self.model_file_name):

            def CustomGetSubSet():
                def inner(data):
                    graph, labels, sets = data

                    randset= np.random.randint(np.max(sets[0][:, 0]) + 1)

                    nodeidxs = np.where(sets[0][:, 0] == randset)[0]
                    edgeidxs = np.where(sets[1][:, 0] == randset)[0]

                    node_features = graph[0][nodeidxs]
                    edge_features = graph[1][edgeidxs]
                    edge_connections = graph[2][edgeidxs] - np.min(nodeidxs)

                    weights = graph[3][edgeidxs]

                    node_labels = labels[0][nodeidxs]
                    edge_labels = labels[1][edgeidxs]
                    glob_labels = labels[2][randset]

                    #node_sets = sets[0][nodeidxs]
                    #edge_sets = sets[1][edgeidxs]

                    #node_sets[:,0] = 0
                    #edge_sets[:,0] = 0

                    frame_sets = sets[2][nodeidxs]

                    return (node_features, edge_features, edge_connections, weights), (
                        node_labels,
                        edge_labels,
                        glob_labels,
                    ), (frame_sets) #, (node_sets, edge_sets, frame_sets)

                return inner

            def CustomGetSubGraph():
                def inner(data):
                    graph, labels, sets = data

                    min_num_nodes = 500
                    max_num_nodes = 750

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

                        nodes_to_select = np.random.choice(nodeidxs, size=number_of_non_clusterized_nodes, replace=False)
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

            def AugmentCentroids(rotate, flip_x, flip_y):
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
                    #>> dt.Lambda(CustomDatasetBalancing)
                    >> dt.Lambda(
                        AugmentCentroids,
                        rotate=lambda: np.random.rand() * 2 * np.pi,
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

        else:
            self.magik_architecture.load_weights(self.model_file_name)

        del train_full_graph

        if not os.path.exists(self.threshold_file_name):
            true = []
            pred = []

            true, pred = self.test_with_datasets_from_path(path, apply_threshold=False, save_result=True)

            count = Counter(true)
            positive_is_majority = count[1] > count[0]

            if positive_is_majority:
                true = 1 - np.array(true)
                pred = 1 - np.array(pred)

            thresholds = np.round(np.arange(0.05,0.95,0.025), 3)

            self.threshold = ghostml.optimize_threshold_from_predictions(true, pred, thresholds, ThOpt_metrics = 'ROC')

            if positive_is_majority:
                self.threshold = 1 - self.threshold
        else:
            with open(self.threshold_file_name, "r") as f:
                self.threshold = float(f.read()) 

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

    def save_model(self):
        self.magik_architecture.save_weights(self.model_file_name)

        with open(self.threshold_file_name, "w") as f:
            f.write(str(self.threshold))
        
    def load_model(self):
        self.build_network()
        self.magik_architecture.load_weights(self.model_file_name)

        with open(self.threshold_file_name, "r") as f:
            self.threshold = float(f.read())
