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
import ghostml
import networkx as nx
from shapely.geometry import MultiPoint, Point
import os
import tqdm
from sklearn.neighbors import NearestNeighbors

from ..utils import *
from ..CONSTANTS import *

class ClusterDetector():
    @classmethod
    def default_hyperparameters(cls):
        """
        partition_size can be equal to None.
        However, this can produce memory limitations
        during training and inference
        """
        return {
            "partition_size": 4000,
            "epochs": 10,
            "number_of_frames_used_in_simulations": 1000,
            "batch_size": 1,
            "training_set_in_epoch_size": 512,
            "ignore_no_clusters_experiments_during_training": False
        }

    @classmethod
    def analysis_hyperparameters(cls):
        return {
            "partition_size": [500,1000,1500,2000,2500,3000,3500,4000]
        }

    def __init__(self, height=10, width=10, static=False):
        self._output_type = "edges"

        self.magik_architecture = None
        self.threshold = 0.5
        self.static = static

        self.hyperparameters = self.__class__.default_hyperparameters()
        self.height = height
        self.width = width

    @property
    def node_features(self):
        return [MAGIK_X_POSITION_COLUMN_NAME, MAGIK_Y_POSITION_COLUMN_NAME]

    @property
    def edge_features(self):
        if self.static:
            return ["distance"]
        else:
            return ["distance", "t-difference"]

    def set_dimensions(self, height, width):
        self.height = height
        self.width = width

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
            optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=['accuracy', tf.keras.metrics.AUC(), positive_rate, negative_rate]
        )

        #self.magik_architecture.summary()

    def transform_smlm_dataset_to_magik_dataframe(self, smlm_dataframe, set_number=0, ignore_non_clustered_localizations=True):
        smlm_dataframe = smlm_dataframe.rename(columns={
            X_POSITION_COLUMN_NAME: MAGIK_X_POSITION_COLUMN_NAME,
            Y_POSITION_COLUMN_NAME: MAGIK_Y_POSITION_COLUMN_NAME,
            CLUSTER_ID_COLUMN_NAME: MAGIK_LABEL_COLUMN_NAME,
            CLUSTER_ID_COLUMN_NAME+"_predicted": MAGIK_LABEL_COLUMN_NAME_PREDICTED,
        })

        if TIME_COLUMN_NAME in smlm_dataframe.columns:
            smlm_dataframe[TIME_COLUMN_NAME] = smlm_dataframe[TIME_COLUMN_NAME] / ((self.hyperparameters['number_of_frames_used_in_simulations'] - 1) * FRAME_RATE)
        else:
            smlm_dataframe[TIME_COLUMN_NAME] = smlm_dataframe[FRAME_COLUMN_NAME] / (self.hyperparameters['number_of_frames_used_in_simulations'] - 1)
            smlm_dataframe['DUMMY'] = 'to_remove'

        if not self.static:
            smlm_dataframe = smlm_dataframe.sort_values(TIME_COLUMN_NAME, ascending=True, inplace=False).reset_index(drop=True)
        else:
            smlm_dataframe = smlm_dataframe.sort_values([MAGIK_X_POSITION_COLUMN_NAME, MAGIK_Y_POSITION_COLUMN_NAME], ascending=[True, True], inplace=False).reset_index(drop=True)

        smlm_dataframe['original_index_for_recovery'] = smlm_dataframe.index

        if ignore_non_clustered_localizations:
            if CLUSTERIZED_COLUMN_NAME+"_predicted" in smlm_dataframe.columns:
                smlm_dataframe = smlm_dataframe[smlm_dataframe[CLUSTERIZED_COLUMN_NAME+"_predicted"] == 1]
            else:
                smlm_dataframe = smlm_dataframe[smlm_dataframe[CLUSTERIZED_COLUMN_NAME] == 1]

        smlm_dataframe = smlm_dataframe.drop(["Unnamed: 0"], axis=1, errors="ignore")
        smlm_dataframe.loc[:, smlm_dataframe.columns.str.contains(MAGIK_POSITION_COLUMN_NAME)] = (smlm_dataframe.loc[:, smlm_dataframe.columns.str.contains(MAGIK_POSITION_COLUMN_NAME)] / np.array([self.width, self.height]))

        smlm_dataframe[MAGIK_DATASET_COLUMN_NAME] = set_number

        if MAGIK_LABEL_COLUMN_NAME in smlm_dataframe.columns:
            smlm_dataframe[MAGIK_LABEL_COLUMN_NAME] = smlm_dataframe[MAGIK_LABEL_COLUMN_NAME].astype(int)
        else:
            smlm_dataframe[MAGIK_LABEL_COLUMN_NAME] = 0

        smlm_dataframe[MAGIK_LABEL_COLUMN_NAME_PREDICTED] = 0

        return smlm_dataframe.reset_index(drop=True)

    def transform_magik_dataframe_to_smlm_dataset(self, magik_dataframe):
        magik_dataframe = magik_dataframe.rename(columns={
            MAGIK_X_POSITION_COLUMN_NAME: X_POSITION_COLUMN_NAME,
            MAGIK_Y_POSITION_COLUMN_NAME: Y_POSITION_COLUMN_NAME,
            MAGIK_LABEL_COLUMN_NAME: CLUSTER_ID_COLUMN_NAME,
            MAGIK_LABEL_COLUMN_NAME_PREDICTED: CLUSTER_ID_COLUMN_NAME+"_predicted",
        })

        if not self.static:
            if 'DUMMY' in magik_dataframe.columns:
                magik_dataframe = magik_dataframe.drop(['DUMMY', TIME_COLUMN_NAME], axis=1, errors="ignore")
            else:
                magik_dataframe[TIME_COLUMN_NAME] = magik_dataframe[TIME_COLUMN_NAME] * ((self.hyperparameters['number_of_frames_used_in_simulations'] - 1) * FRAME_RATE)

        magik_dataframe.loc[:, X_POSITION_COLUMN_NAME] = (magik_dataframe.loc[:, X_POSITION_COLUMN_NAME] * np.array([self.width]))
        magik_dataframe.loc[:, Y_POSITION_COLUMN_NAME] = (magik_dataframe.loc[:, Y_POSITION_COLUMN_NAME] * np.array([self.height]))

        magik_dataframe = magik_dataframe.drop(MAGIK_DATASET_COLUMN_NAME, axis=1)

        return magik_dataframe.reset_index(drop=True)

    def get_dataset_from_path(self, path, set_number=0, ignore_non_clustered_localizations=True):
        return self.transform_smlm_dataset_to_magik_dataframe(pd.read_csv(path), set_number=set_number, ignore_non_clustered_localizations=ignore_non_clustered_localizations)

    def get_dataset_file_paths_from(self, path):
        if not self.static:
            return [os.path.join(path, file_name) for file_name in os.listdir(path) if file_name.endswith(".csv") and len(file_name.split('.'))==2]
        else:
            return [os.path.join(path, file_name) for file_name in os.listdir(path) if file_name.endswith(".tsv.csv")]

    def get_datasets_from_path(self, path, ignore_non_clustered_localizations=True, ignore_non_clustered_experiments=False):
        """
        This method is different of the Localization Classifier method
        because there are some datasets for testing that have no clusters
        """
        full_dataset = pd.DataFrame({})
        set_index = 0

        for csv_file_path in self.get_dataset_file_paths_from(path):
            set_dataframe = self.get_dataset_from_path(csv_file_path, set_number=set_index, ignore_non_clustered_localizations=ignore_non_clustered_localizations)

            if not set_dataframe.empty and (not ignore_non_clustered_experiments or len(set_dataframe[set_dataframe[CLUSTERIZED_COLUMN_NAME] == 1]) != 0):
                full_dataset = pd.concat([full_dataset, set_dataframe], ignore_index=True)
                set_index += 1

        return full_dataset.reset_index(drop=True)

    def predict(self, magik_dataset, apply_threshold=True, detect_clusters=True, verbose=True, original_dataset_path=None, suppose_perfect_classification=False, grapht=None, real_edges_weights=None):
        assert not(not apply_threshold and detect_clusters), 'Cluster Detection cannot be performed without threshold apply'

        if len(magik_dataset) != 0:
            magik_dataset = magik_dataset.copy()
            
            if grapht is None and real_edges_weights is None:
                grapht, real_edges_weights = self.build_graph(magik_dataset, verbose=False, return_real_edges_weights=True)

            predictions = np.empty((len(grapht[0][1]), 1))

            number_of_edges = len(grapht[0][2])
            if self.hyperparameters['partition_size'] is not None:
                partitions_initial_index = list(range(0,number_of_edges,self.hyperparameters['partition_size']))
            else:
                partitions_initial_index = list(range(0,number_of_edges,number_of_edges))
            if not suppose_perfect_classification:
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

                    v = [
                        np.expand_dims(considered_nodes_features, 0),
                        np.expand_dims(considered_edges_features, 0),
                        np.expand_dims(considered_edges, 0),
                        np.expand_dims(considered_edges_weights, 0),
                    ]

                    with get_device():
                        raw_predictions = self.magik_architecture(v).numpy()
                        predictions[initial_index:final_index] = (raw_predictions > self.threshold)[0, ...] if apply_threshold else (raw_predictions)[0, ...]
            else:
                predictions = grapht[1][1]

            if not detect_clusters:
                return grapht[1][1], predictions

            edges_to_remove = np.where(predictions == 0)[0]
            remaining_edges_keep = np.delete(grapht[0][2], edges_to_remove, axis=0)

            if len(remaining_edges_keep) != 0:
                #remaining_edges_weights = np.expand_dims(np.delete(grapht[0][1][:, 0], edges_to_remove, axis=0), -1) #Spatial Distance Weight
                remaining_edges_weights = np.expand_dims(np.delete(real_edges_weights, edges_to_remove, axis=0), -1) #Real Distance Weight
                remaining_edges_weights = 1 / remaining_edges_weights #Inverse Distance Weight

                G=nx.Graph()
                G.add_weighted_edges_from(np.hstack((remaining_edges_keep, remaining_edges_weights)))  #Weighted Graph

                if suppose_perfect_classification:
                    #Connected Components
                    cluster_sets = nx.connected_components(G)
                else:
                    #Louvain Method with Weights
                    cluster_sets = nx.community.louvain_communities(G, weight='weight')

                    """
                    #Louvain Method without Weights
                    cluster_sets = nx.community.louvain_communities(G, weight=None)
                    """

                    """
                    #Greedy Modularity with Weights
                    cluster_sets = nx.community.greedy_modularity_communities(G, weight='weight')
                    """

                    """
                    #Greedy Modularity without Weights
                    cluster_sets = nx.community.greedy_modularity_communities(G, weight=None)
                    """

                magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED] = 0

                for index, a_set in enumerate(cluster_sets):
                    for value in a_set:
                        magik_dataset.loc[value, MAGIK_LABEL_COLUMN_NAME_PREDICTED] = index + 1

                magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED] = magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED].astype(int)

                if MAGIK_LABEL_COLUMN_NAME in magik_dataset.columns:
                    magik_dataset[MAGIK_LABEL_COLUMN_NAME] = magik_dataset[MAGIK_LABEL_COLUMN_NAME].astype(int)

                #Last Correction
                if not self.static and not suppose_perfect_classification:
                    cluster_indexes_list = magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED].unique().tolist()
                    cluster_id_index = 0

                    if 0 in cluster_indexes_list:
                        cluster_indexes_list.remove(0)

                    while cluster_id_index < len(cluster_indexes_list):
                        cluster_id = cluster_indexes_list[cluster_id_index]
                        cluster_dataframe = magik_dataset[magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED] == cluster_id].copy()

                        if len(cluster_dataframe) > 0:
                            cluster_polygon = MultiPoint([point for point in cluster_dataframe[[MAGIK_X_POSITION_COLUMN_NAME, MAGIK_Y_POSITION_COLUMN_NAME]].values.tolist()]).convex_hull
                            cluster_centroid = np.mean(cluster_dataframe[[MAGIK_X_POSITION_COLUMN_NAME, MAGIK_Y_POSITION_COLUMN_NAME]].values, axis=0)
                            last_localizations_cluster_dataframe = cluster_dataframe[cluster_dataframe[TIME_COLUMN_NAME] == cluster_dataframe[TIME_COLUMN_NAME].max()].copy()

                            if len(last_localizations_cluster_dataframe) == 1:
                                last_localization = last_localizations_cluster_dataframe[[MAGIK_X_POSITION_COLUMN_NAME, MAGIK_Y_POSITION_COLUMN_NAME, TIME_COLUMN_NAME]].values.tolist()[0]
                            else:
                                last_localizations = last_localizations_cluster_dataframe[[MAGIK_X_POSITION_COLUMN_NAME, MAGIK_Y_POSITION_COLUMN_NAME, TIME_COLUMN_NAME]].values.tolist()
                                centroid_distances = []

                                for last_localization in last_localizations:
                                    x = cluster_centroid[0] - last_localization[0]
                                    y = cluster_centroid[1] - last_localization[1]

                                    centroid_distances.append(((x**2)+(y**2))**(1/2))

                                last_localization = last_localizations[np.argmin(centroid_distances)]

                            other_clusters = magik_dataset[magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED] != 0].copy()
                            other_clusters = other_clusters[other_clusters[MAGIK_LABEL_COLUMN_NAME_PREDICTED] != cluster_id].copy()

                            nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1, algorithm='kd_tree').fit(other_clusters[[MAGIK_X_POSITION_COLUMN_NAME, MAGIK_Y_POSITION_COLUMN_NAME, TIME_COLUMN_NAME]].values)
                            indices = nbrs.kneighbors([last_localization], return_distance=False)
                            cluster_candidate_id = other_clusters.iloc[indices[0][0]][MAGIK_LABEL_COLUMN_NAME_PREDICTED]
                            candidate_cluster_dataframe = magik_dataset[magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED] == cluster_candidate_id].copy()

                            candidate_cluster_centroid = np.mean(candidate_cluster_dataframe[[MAGIK_X_POSITION_COLUMN_NAME, MAGIK_Y_POSITION_COLUMN_NAME]].values, axis=0)

                            if cluster_polygon.contains(Point(candidate_cluster_centroid[0], candidate_cluster_centroid[1])):
                                magik_dataset.loc[magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED] == cluster_candidate_id, MAGIK_LABEL_COLUMN_NAME_PREDICTED] = cluster_id
                                cluster_indexes_list.remove(cluster_candidate_id)
                            else:
                                cluster_id_index += 1

                    cluster_indexes_list = list(set(magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED]))

                    if 0 in cluster_indexes_list:
                        cluster_indexes_list.remove(0)

                    for cluster_index in cluster_indexes_list:
                        cluster_dataframe = magik_dataset[ magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED] == cluster_index ]

                        if not (len(cluster_dataframe) >= 5 and cluster_dataframe[TIME_COLUMN_NAME].max() - cluster_dataframe[TIME_COLUMN_NAME].min() > FRAME_RATE):
                            magik_dataset.loc[cluster_dataframe.index, MAGIK_LABEL_COLUMN_NAME_PREDICTED] = 0
                elif self.static and not suppose_perfect_classification:
                    #This code needs to be improved. We noticed that this improved performance in static datasets
                    boolean_dataframe = magik_dataset.groupby(MAGIK_LABEL_COLUMN_NAME_PREDICTED).count()[MAGIK_X_POSITION_COLUMN_NAME] >= 5
                    for i in boolean_dataframe.index:
                        if not boolean_dataframe.loc[i]:
                            magik_dataset[magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED]==i] = 0

            else:
                magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED] = 0

        #Filtered Localization Reinsertion
        if original_dataset_path is not None:
            original_dataset = self.transform_smlm_dataset_to_magik_dataframe(pd.read_csv(original_dataset_path), ignore_non_clustered_localizations=False)

            for _, row in magik_dataset.iterrows():
                original_dataset.loc[original_dataset["original_index_for_recovery"] == row["original_index_for_recovery"], MAGIK_LABEL_COLUMN_NAME_PREDICTED] = row[MAGIK_LABEL_COLUMN_NAME_PREDICTED]

            magik_dataset = original_dataset

        magik_dataset[CLUSTERIZED_COLUMN_NAME+'_predicted'] = (magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED] != 0).astype(int)

        return magik_dataset

    def build_graph(self, full_nodes_dataset, verbose=True, return_real_edges_weights=False):        
        edges_dataframe = pd.DataFrame({
            "distance": [],
            "index_1": [],
            "index_2": [],
            MAGIK_DATASET_COLUMN_NAME: [],
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

            if len(df_window) < 4:
                list_of_edges = []
            else:
                list_of_edges = delaunay_from_dataframe(df_window, [MAGIK_X_POSITION_COLUMN_NAME, MAGIK_Y_POSITION_COLUMN_NAME])

                if not self.static:
                    list_of_edges += delaunay_from_dataframe(df_window, [MAGIK_X_POSITION_COLUMN_NAME, TIME_COLUMN_NAME])
                    list_of_edges += delaunay_from_dataframe(df_window, [MAGIK_Y_POSITION_COLUMN_NAME, TIME_COLUMN_NAME])
                    list_of_edges += delaunay_from_dataframe(df_window, [MAGIK_X_POSITION_COLUMN_NAME, MAGIK_Y_POSITION_COLUMN_NAME, TIME_COLUMN_NAME])

                new_index_to_old_index = {new_index:df_window.loc[new_index, 'index'] for new_index in df_window.index.values}
                list_of_edges = np.vectorize(new_index_to_old_index.get)(list_of_edges)
                list_of_edges = np.unique(list_of_edges, axis=0).tolist() # remove duplicates
            
            list_of_edges_as_dataframe = pd.DataFrame({'index_x': [edge[0] for edge in list_of_edges], 'index_y': [edge[1] for edge in list_of_edges]})
            simplified_cross = list_of_edges_as_dataframe.merge(df_window.rename(columns={old_column_name: old_column_name+'_x' for old_column_name in df_window.columns}), on='index_x')
            simplified_cross = simplified_cross.merge(df_window.rename(columns={old_column_name: old_column_name+'_y' for old_column_name in df_window.columns}), on='index_y')

            df_window = simplified_cross.copy()
            df_window = df_window[df_window['index_x'] != df_window['index_y']]
            df_window['distance-x'] = df_window[f"{MAGIK_X_POSITION_COLUMN_NAME}_x"] - df_window[f"{MAGIK_X_POSITION_COLUMN_NAME}_y"]
            df_window['distance-y'] = df_window[f"{MAGIK_Y_POSITION_COLUMN_NAME}_x"] - df_window[f"{MAGIK_Y_POSITION_COLUMN_NAME}_y"]
            df_window['t-difference'] = df_window[f"{TIME_COLUMN_NAME}_x"] - df_window[f"{TIME_COLUMN_NAME}_y"]
            df_window['distance'] = ((df_window['distance-x']**2) + (df_window['distance-y']**2))**(1/2)
            df_window['real_distance'] = ((df_window['distance-x']**2) + (df_window['distance-y']**2) + (df_window['t-difference']**2))**(1/2)

            same_cluster_series_boolean = (df_window[MAGIK_LABEL_COLUMN_NAME+"_x"] == df_window[MAGIK_LABEL_COLUMN_NAME+"_y"])
            x_index_is_clustered_series_boolean = (df_window[MAGIK_LABEL_COLUMN_NAME+"_x"] != 0)
            y_index_is_clustered_series_boolean = (df_window[MAGIK_LABEL_COLUMN_NAME+"_y"] != 0)
            there_are_only_clustered_localization_series_boolean = x_index_is_clustered_series_boolean & y_index_is_clustered_series_boolean
            df_window['same_cluster'] = (same_cluster_series_boolean & there_are_only_clustered_localization_series_boolean)

            edges = [sorted(edge) for edge in df_window[["index_x", "index_y"]].values.tolist()]

            new_edges_dataframe = pd.concat([new_edges_dataframe, pd.DataFrame({
                'index_1': [edge[0] for edge in edges],
                'index_2': [edge[1] for edge in edges],
                'distance': [value[0] for value in df_window[["distance"]].values.tolist()],
                't-difference': [abs(value[0]) for value in df_window[["t-difference"]].values.tolist()],
                'real_distance': [value[0] for value in df_window[["real_distance"]].values.tolist()],
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

        grapht = (
                (nodefeatures, edgefeatures, sparseadjmtx, edgeweights),
                (nfsolution, efsolution, global_property),
                (nodesets, edgesets, framesets)
            )

        if return_real_edges_weights:
            return grapht, edges_dataframe['real_distance'].to_numpy()
        else:
            return grapht

    @property
    def train_full_graph_file_name(self):
        return ('static_' if self.static else "") + f"edge_classifier.tmp"

    @property
    def model_file_name(self):
        return ('static_' if self.static else "") + f"edge_classifier_batch_size_{self.hyperparameters['batch_size']}_partition_{self.hyperparameters['partition_size']}.h5"

    @property
    def predictions_file_name(self):
        return ('static_' if self.static else "") + f"edge_classifier_batch_size_{self.hyperparameters['batch_size']}_partition_{self.hyperparameters['partition_size']}.csv"
    
    @property
    def history_training_info_file_name(self):
        return ('static_' if self.static else "") + f"edge_classifier_batch_size_{self.hyperparameters['batch_size']}_partition_{self.hyperparameters['partition_size']}.json"

    @property
    def threshold_file_name(self):
        return ('static_' if self.static else "") + f"edge_classifier_batch_size_{self.hyperparameters['batch_size']}_partition_{self.hyperparameters['partition_size']}.bin"

    def test_with_datasets_from_path(self, path, plot=False, apply_threshold=True, save_result=False, save_predictions=False, verbose=True, detect_clusters=False, check_if_predictions_file_name_exists=False, ignore_non_clustered_localizations=True):
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
                if not self.get_dataset_from_path(csv_file_name, ignore_non_clustered_localizations=ignore_non_clustered_localizations).empty:
                    if detect_clusters:
                        predictions = self.predict(self.get_dataset_from_path(csv_file_name, ignore_non_clustered_localizations=ignore_non_clustered_localizations), apply_threshold=True, verbose=False, detect_clusters=True, original_dataset_path=csv_file_name)
                        if save_predictions:
                            predictions.to_csv(csv_file_name+f"_predicted_with_batch_size_{self.hyperparameters['batch_size']}_partition_{self.hyperparameters['partition_size']}.csv", index=False)
                    else:
                        result = self.predict(self.get_dataset_from_path(csv_file_name, ignore_non_clustered_localizations=ignore_non_clustered_localizations), apply_threshold=apply_threshold, detect_clusters=False, verbose=False)
                        true += result[0][:,0].tolist()
                        pred += result[1][:,0].tolist()

            if save_result:
                pd.DataFrame({
                    'true': true,
                    'pred': pred
                }).to_csv(self.predictions_file_name, index=False)

        if plot:
            self.plot_confusion_matrix(true, pred)

        if not detect_clusters:
            return true, pred

    def fit_with_datasets_from_path(self, path, save_checkpoints=False):

        if self.load_keras_model() is None:
            if os.path.exists(self.train_full_graph_file_name):
                fileObj = open(self.train_full_graph_file_name, 'rb')
                train_full_graph = pickle.load(fileObj)
                fileObj.close()
            else:
                ignoring_non_clustered_localizations = self.get_datasets_from_path(path, ignore_non_clustered_localizations=True, ignore_non_clustered_experiments=self.hyperparameters["ignore_no_clusters_experiments_during_training"])

                number_of_sets_ignoring = len(set(ignoring_non_clustered_localizations['set'].values.tolist()))
                not_ignoring_non_clustered_localizations = self.get_datasets_from_path(path, ignore_non_clustered_localizations=False, ignore_non_clustered_experiments=self.hyperparameters["ignore_no_clusters_experiments_during_training"])
                not_ignoring_non_clustered_localizations['set'] += number_of_sets_ignoring

                dataframe_combined = pd.concat([ignoring_non_clustered_localizations, not_ignoring_non_clustered_localizations], ignore_index=True)

                train_full_graph = self.build_graph(dataframe_combined)

                fileObj = open(self.train_full_graph_file_name, 'wb')
                pickle.dump(train_full_graph, fileObj)
                fileObj.close()

            def CustomGetFeature(full_graph, **kwargs):
                return (
                    dt.Value(full_graph)
                    >> dt.Lambda(CustomGetSubSet,
                        ignore_non_cluster_experiments= lambda: False
                    )
                    >> dt.Lambda(CustomGetSubGraphByNumberOfEdges,
                        min_num_edges=lambda: self.hyperparameters["partition_size"],
                        max_num_edges=lambda: self.hyperparameters["partition_size"]
                    )
                    #>> dt.Lambda(CustomEdgeBalancing)
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

            with get_device():
                with generator:
                    self.history_training_info = self.magik_architecture.fit(generator, epochs=self.hyperparameters["epochs"]).history

            self.save_history_training_info()

            if save_checkpoints:
                self.save_keras_model()

            del generator
            del train_full_graph

        if self.load_threshold() is None:
            """
            print("Running Ghost...")

            true, pred = self.test_with_datasets_from_path(path, apply_threshold=False, save_result=False, verbose=True, ignore_non_clustered_localizations=False)

            count = Counter(true)
            positive_is_majority = count[1] > count[0]

            if positive_is_majority:
                true = 1 - np.array(true)
                pred = 1 - np.array(pred)

            thresholds = np.round(np.arange(0.05,0.95,0.025), 3)

            self.threshold = ghostml.optimize_threshold_from_predictions(true, pred, thresholds, ThOpt_metrics = 'ROC', N_subsets=1, subsets_size=0.00001, with_replacement=False)
            """
            self.threshold = 0.5

            if save_checkpoints:
                self.save_threshold()

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
        self.save_threshold()

    def save_threshold(self):
        save_number_in_file(self.threshold_file_name, self.threshold)

    def save_keras_model(self):
        self.magik_architecture.save_weights(self.model_file_name)
        
    def load_keras_model(self, file_path=None, raise_error=False):
        try:
            self.build_network()
            selected_file_path = self.model_file_name if file_path is None else file_path
            self.magik_architecture.load_weights(selected_file_path)
        except FileNotFoundError as e:
            if raise_error:
                raise e
            else:
                print(f"WARNING: {self} has not found keras model file (file name:{selected_file_path})")
            return None

        return self.magik_architecture

    def load_threshold(self, file_path=None, raise_error=False):
        selected_file_path = self.threshold_file_name if file_path is None else file_path

        try:
            self.threshold = read_number_from_file(selected_file_path)
        except ValueError:
            raise Exception('Threshold file should only contain a float number.')
        if self.threshold is None:
            if raise_error:
                raise Exception('Threshold file not found')
            else:
                print(f"WARNING: {self} has not found threshold file (file name:{selected_file_path})")
            return None

        return self.threshold

    def load_model(self, keras_model_file_path=None, threshold_file_path=None, raise_error=False):
        self.load_keras_model(file_path=keras_model_file_path, raise_error=raise_error)
        self.load_threshold(file_path=threshold_file_path, raise_error=raise_error)
