import os
import tqdm
import pickle
import json

from deeptrack.models.gnns.generators import ContinuousGraphGenerator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import deeptrack as dt
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
import networkx as nx

from training_utils import *
from CONSTANTS import *


class ClusterEdgeRemover():
    @classmethod
    def default_hyperparameters(cls):
        return {
            "learning_rate": 0.001,
            "partition_size": 10000,
            "epochs": 25,
            "batch_size": 1,
            "training_set_in_epoch_size": 25
        }

    @classmethod
    def analysis_hyperparameters(cls):
        return {
            #"learning_rate": [0.1, 0.01, 0.001],
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
            edge_output_activation="sigmoid",
            output_type=self._output_type,
        )

        self.magik_architecture.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.hyperparameters['learning_rate']),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )

        self.magik_architecture.summary()

    def transform_smlm_dataset_to_magik_dataframe(self, smlm_dataframe, set_number=0, ignored_non_clustered_localizations=True):
        smlm_dataframe = smlm_dataframe.rename(columns={
            X_POSITION_COLUMN_NAME: MAGIK_X_POSITION_COLUMN_NAME,
            Y_POSITION_COLUMN_NAME: MAGIK_Y_POSITION_COLUMN_NAME,
            CLUSTER_ID_COLUMN_NAME: MAGIK_LABEL_COLUMN_NAME,
            CLUSTER_ID_COLUMN_NAME+"_predicted": MAGIK_LABEL_COLUMN_NAME_PREDICTED,
        })

        smlm_dataframe['original_index_for_recovery'] = smlm_dataframe.index

        if ignored_non_clustered_localizations:
            if 'clusterized_predicted' in smlm_dataframe.columns:
                smlm_dataframe = smlm_dataframe[smlm_dataframe['clusterized_predicted'] == 1]
            else:
                smlm_dataframe = smlm_dataframe[smlm_dataframe['clusterized'] == 1]

        smlm_dataframe = smlm_dataframe.drop([CLUSTERIZED_COLUMN_NAME, CLUSTERIZED_COLUMN_NAME+'_predicted', PARTICLE_ID_COLUMN_NAME, "Unnamed: 0"], axis=1, errors="ignore")
        smlm_dataframe.loc[:, smlm_dataframe.columns.str.contains(MAGIK_POSITION_COLUMN_NAME)] = (smlm_dataframe.loc[:, smlm_dataframe.columns.str.contains(MAGIK_POSITION_COLUMN_NAME)] / np.array([self.width, self.height]))
        
        smlm_dataframe[TIME_COLUMN_NAME] = smlm_dataframe[TIME_COLUMN_NAME] / smlm_dataframe[TIME_COLUMN_NAME].abs().max()
        smlm_dataframe[MAGIK_DATASET_COLUMN_NAME] = set_number
        smlm_dataframe[MAGIK_LABEL_COLUMN_NAME] = smlm_dataframe[MAGIK_LABEL_COLUMN_NAME].astype(int)
        smlm_dataframe[MAGIK_LABEL_COLUMN_NAME_PREDICTED] = 0

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
                full_dataset = pd.concat([full_dataset, set_dataframe], ignore_index=True)
                set_index += 1

        return full_dataset.reset_index(drop=True)

    def predict(self, magik_dataset, apply_threshold=True, detect_clusters=True, verbose=True, original_dataset_path=None):
        assert not(not apply_threshold and detect_clusters), 'Cluster Detection cannot be performed without threshold apply'

        magik_dataset = magik_dataset.copy()

        grapht = self.build_graph(magik_dataset, for_predict=True, verbose=False)

        predictions = np.empty((len(grapht[0][1]), 1))

        number_of_edges = len(grapht[0][2])
        partitions_initial_index = list(range(0,number_of_edges,self.hyperparameters['partition_size']))
        
        for index, initial_index in enumerate(partitions_initial_index):
            if index == len(partitions_initial_index)-1:
                final_index = number_of_edges
            else:
                final_index = partitions_initial_index[index+1]

            considered_edges_features = grapht[0][1][initial_index:final_index]
            considered_edges = grapht[0][2][initial_index:final_index]
            considered_edges_weights = grapht[0][3][initial_index:final_index]

            considered_nodes = np.unique(considered_edges.flatten())
            considered_nodes_features = grapht[0][0][considered_nodes]

            old_index_to_new_index = {old_index:new_index for new_index, old_index in enumerate(considered_nodes)}
            considered_edges = np.vectorize(old_index_to_new_index.get)(considered_edges)
            considered_edges = np.unique(considered_edges, axis=0) # remove duplicates

            v = [
                considered_nodes_features.reshape(1, considered_nodes_features.shape[0], considered_nodes_features.shape[1]),
                considered_edges_features.reshape(1, considered_edges_features.shape[0], considered_edges_features.shape[1]),
                considered_edges.reshape(1, considered_edges.shape[0], considered_edges.shape[1]),
                considered_edges_weights.reshape(1, considered_edges_weights.shape[0], considered_edges_weights.shape[1]),
            ]

            with get_device():
                predictions[initial_index:final_index] = (self.magik_architecture(v).numpy() > self.threshold)[0, ...] if apply_threshold else (self.magik_architecture(v).numpy())[0, ...]

        if not detect_clusters:
            return grapht[1][1], predictions

        edges_to_remove = np.where(predictions == 0)[0]
        remaining_edges_keep = np.delete(grapht[0][2], edges_to_remove, axis=0)
        remaining_edges_weights = np.delete(grapht[0][1], edges_to_remove, axis=0)

        """
        As the cluster detection is sensible to the edge pruning (if only one edge is misclassified as positive, two clusters are merged),
        detected connected components should be segmented to avoid this problem. If two clusters are merged and the performance of the
        edge classifier is too high, both clusters may be segmentated maximizing the modularity of graph partition. Hint: Levounian
        """
        cluster_sets = []

        G=nx.Graph()
        G.add_edges_from(remaining_edges_keep)
        #G.add_weighted_edges_from(np.hstack((remaining_edges_keep, remaining_edges_weights)))

        #S = [G.subgraph(c).copy() for c in nx.connected_components(G)]

        #for subset in S:
        #    cluster_sets += nx.community.louvain_communities(subset)

        cluster_sets = nx.community.louvain_communities(G)
        #cluster_sets = nx.connected_components(G)

        """
        im = Infomap(two_level=True, silent=True, num_trials=20)
        im.add_networkx_graph(G)
        im.run()

        modules = im.get_modules()
        cluster_sets = [set() for _ in range(len(set(modules.values())))]
        [cluster_sets[c - 1].add(n) for n, c in modules.items()]
        """

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
        """

        magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED] = 0

        for index, a_set in enumerate(cluster_sets):
            for value in a_set:
                magik_dataset.loc[value, MAGIK_LABEL_COLUMN_NAME_PREDICTED] = index + 1

        magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED] = magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED].astype(int)

        if MAGIK_LABEL_COLUMN_NAME in magik_dataset.columns:
            magik_dataset[MAGIK_LABEL_COLUMN_NAME] = magik_dataset[MAGIK_LABEL_COLUMN_NAME].astype(int)

        if original_dataset_path is not None:
            original_dataset = self.transform_smlm_dataset_to_magik_dataframe(pd.read_csv(original_dataset_path), ignored_non_clustered_localizations=False)

            for _, row in magik_dataset.iterrows():
                original_dataset.loc[row["original_index_for_recovery"], MAGIK_LABEL_COLUMN_NAME_PREDICTED] = row[MAGIK_LABEL_COLUMN_NAME_PREDICTED]

            magik_dataset = original_dataset

        return magik_dataset

    def build_graph(self, full_nodes_dataset, verbose=True, for_predict=False):
        edges_dataframe = pd.DataFrame({
            "distance": [],
            "index_1": [],
            "index_2": [],
            "set": [],
            "same_cluster": []
        })

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
            list_of_dataframes = []

            for edge in list_of_edges:
                x_index = df_window["index"] == edge[0]
                y_index = df_window["index"] == edge[1]

                list_of_dataframes.append(pd.DataFrame({
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
                }))

            simplified_cross = pd.concat(list_of_dataframes, ignore_index=True)

            df_window = simplified_cross.copy()
            df_window = df_window[df_window['index_x'] != df_window['index_y']]
            df_window['distance-x'] = df_window[f"{MAGIK_X_POSITION_COLUMN_NAME}_x"] - df_window[f"{MAGIK_X_POSITION_COLUMN_NAME}_y"]
            df_window['distance-y'] = df_window[f"{MAGIK_Y_POSITION_COLUMN_NAME}_x"] - df_window[f"{MAGIK_Y_POSITION_COLUMN_NAME}_y"]
            df_window['distance'] = ((df_window['distance-x']**2) + (df_window['distance-y']**2))**(1/2)
            df_window['same_cluster'] = (df_window[MAGIK_LABEL_COLUMN_NAME+"_x"] == df_window[MAGIK_LABEL_COLUMN_NAME+"_y"])

            if for_predict or not df_window['same_cluster'].all():
                edges = [sorted(edge) for edge in df_window[["index_x", "index_y"]].values.tolist()]
                
                new_edges_dataframe = pd.concat([new_edges_dataframe, pd.DataFrame({
                    'index_1': [edge[0] for edge in edges],
                    'index_2': [edge[1] for edge in edges],
                    'distance': [value[0] for value in df_window[["distance"]].values.tolist()],
                    'same_cluster': [value[0] for value in df_window[["same_cluster"]].values.tolist()],
                })], ignore_index=True)

            new_edges_dataframe = new_edges_dataframe.drop_duplicates()
            new_edges_dataframe['set'] = setid

            edges_dataframe = pd.concat([edges_dataframe, new_edges_dataframe], ignore_index=True)

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
    def predictions_file_name(self):
        return f"edge_classifier_batch_size_{self.hyperparameters['batch_size']}_partition_{self.hyperparameters['partition_size']}.csv"
    
    @property
    def history_training_info_file_name(self):
        return f"edge_classifier_batch_size_{self.hyperparameters['batch_size']}.json"

    def test_with_datasets_from_path(self, path, plot=False, apply_threshold=True, save_result=False, save_predictions=False, verbose=True, detect_clusters=False, check_if_predictions_file_name_exists=False):
        assert not (save_predictions and not detect_clusters)
        assert not (save_result and detect_clusters)

        if check_if_predictions_file_name_exists and os.path.exists(self.predictions_file_name):
            dataframe = pd.read_csv(self.predictions_file_name)
            true, pred = dataframe['true'].values.tolist(), dataframe['pred'].values.tolist()
        else:
            true = []
            pred = []

            iterator = tqdm.tqdm(self.get_dataset_file_paths_from(path)) if verbose else self.get_dataset_file_paths_from(path)

            for csv_file_name in iterator:
                if not self.get_dataset_from_path(csv_file_name).empty:
                    if detect_clusters:
                        predictions = self.predict(self.get_dataset_from_path(csv_file_name), apply_threshold=True, verbose=False, detect_clusters=True, original_dataset_path=csv_file_name)
                        if save_predictions:
                            predictions.to_csv(csv_file_name+f"_predicted_with_batch_size_{self.hyperparameters['batch_size']}_partition_{self.hyperparameters['partition_size']}.csv", index=False)
                    else:
                        result = self.predict(self.get_dataset_from_path(csv_file_name), apply_threshold=apply_threshold, detect_clusters=False, verbose=False)
                        true += result[0][:,0].tolist()
                        pred += result[1][:,0].tolist()

            if save_result:
                pd.DataFrame({
                    'true': true,
                    'pred': pred
                }).to_csv(self.predictions_file_name, index=False)

        if plot:
            raise NotImplementedError("Plotting during testing is not implemented yet for Cluster Edge Remover")

        if not detect_clusters:
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

        if self.load_keras_model() is None:
            def CustomGetFeature(full_graph, **kwargs):
                return (
                    dt.Value(full_graph)
                    >> dt.Lambda(CustomGetSubSet)
                    >> dt.Lambda(CustomGetSubGraph)
                    >> dt.Lambda(CustomEdgeBalancing)
                    >> dt.Lambda(
                        CustomAugmentCentroids,
                        rotate=lambda: np.random.rand() * 2 * np.pi,
                        translate=lambda: np.random.randn(2) * 0,
                        flip_x=lambda: np.random.randint(2),
                        flip_y=lambda: np.random.randint(2)
                    )
                )

            magik_variables = dt.DummyFeature(
                radius=None,
                output_type=self._output_type,
                nofframes=None
            )

            args = {
                "batch_function": lambda graph: graph[0],
                "label_function": lambda graph: graph[1],
                "min_data_size": self.hyperparameters["training_set_in_epoch_size"],
                "batch_size": self.hyperparameters["batch_size"],
                "use_multi_inputs": False,
                **magik_variables.properties(),
            }

            generator = ContinuousGraphGenerator(CustomGetFeature(train_full_graph, **magik_variables.properties()), **args)

            with generator:
                self.magik_architecture.fit(generator, epochs=self.hyperparameters["epochs"])

            self.save_history_training_info()

            del generator
        
        del train_full_graph

    def plot_confusion_matrix(self, ground_truth, Y_predicted, normalized=True):
        confusion_mat = confusion_matrix(y_true=ground_truth, y_pred=Y_predicted)

        if normalized:
            confusion_mat = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]

        labels = ["Not Same Cluster", "Same Cluster"]

        confusion_matrix_dataframe = pd.DataFrame(data=confusion_mat, index=labels, columns=labels)
        sns.set(font_scale=1.5)
        color_map = sns.color_palette(palette="Blues", n_colors=7)
        sns.heatmap(data=confusion_matrix_dataframe, annot=True, annot_kws={"size": 15}, cmap=color_map)

        plt.title(f'Confusion Matrix for Edge Classifier')
        plt.rcParams.update({'font.size': 15})
        plt.ylabel("Ground truth", fontsize=15)
        plt.xlabel("Predicted label", fontsize=15)
        plt.show()

    def save_history_training_info(self):
        with open(self.history_training_info_file_name, "w") as json_file:
            json.dump(self.history_training_info, json_file)

    def save_model(self):
        self.save_keras_model()

    def save_keras_model(self):
        self.magik_architecture.save_weights(self.model_file_name)
        
    def load_keras_model(self):
        try:
            self.build_network()
            self.magik_architecture.load_weights(self.model_file_name)
        except FileNotFoundError:
            return None

        return self.magik_architecture

    def load_model(self):
        self.load_keras_model()
