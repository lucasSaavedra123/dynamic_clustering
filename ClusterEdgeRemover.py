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
import ghostml
from sklearn.cluster import AgglomerativeClustering
import alphashape
from shapely.geometry import Point


from training_utils import *
from CONSTANTS import *


class ClusterEdgeRemover():
    @classmethod
    def default_hyperparameters(cls):
        return {
            "partition_size": 4000,
            "epochs": 10,
            "number_of_frames_used_in_simulations": 1000,
            "batch_size": 1,
            "training_set_in_epoch_size": 512
        }

    @classmethod
    def analysis_hyperparameters(cls):
        return {
            "partition_size": [500,1000,1500,2000,2500,3000,3500,4000]
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

    def transform_smlm_dataset_to_magik_dataframe(self, smlm_dataframe, set_number=0, ignored_non_clustered_localizations=True):
        smlm_dataframe = smlm_dataframe.rename(columns={
            X_POSITION_COLUMN_NAME: MAGIK_X_POSITION_COLUMN_NAME,
            Y_POSITION_COLUMN_NAME: MAGIK_Y_POSITION_COLUMN_NAME,
            CLUSTER_ID_COLUMN_NAME: MAGIK_LABEL_COLUMN_NAME,
            CLUSTER_ID_COLUMN_NAME+"_predicted": MAGIK_LABEL_COLUMN_NAME_PREDICTED,
        })

        smlm_dataframe = smlm_dataframe.sort_values(TIME_COLUMN_NAME, ascending=True, inplace=False).reset_index(drop=True)
        smlm_dataframe['original_index_for_recovery'] = smlm_dataframe.index

        if ignored_non_clustered_localizations:
            if 'clusterized_predicted' in smlm_dataframe.columns:
                smlm_dataframe = smlm_dataframe[smlm_dataframe['clusterized_predicted'] == 1]
            else:
                smlm_dataframe = smlm_dataframe[smlm_dataframe['clusterized'] == 1]

        smlm_dataframe = smlm_dataframe.drop(["Unnamed: 0"], axis=1, errors="ignore")
        smlm_dataframe.loc[:, smlm_dataframe.columns.str.contains(MAGIK_POSITION_COLUMN_NAME)] = (smlm_dataframe.loc[:, smlm_dataframe.columns.str.contains(MAGIK_POSITION_COLUMN_NAME)] / np.array([self.width, self.height]))
        smlm_dataframe[TIME_COLUMN_NAME] = smlm_dataframe[TIME_COLUMN_NAME] / ((self.hyperparameters['number_of_frames_used_in_simulations'] - 1) * FRAME_RATE)
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

        magik_dataframe.loc[:, X_POSITION_COLUMN_NAME] = (magik_dataframe.loc[:, X_POSITION_COLUMN_NAME] * np.array([self.width]))
        magik_dataframe.loc[:, Y_POSITION_COLUMN_NAME] = (magik_dataframe.loc[:, Y_POSITION_COLUMN_NAME] * np.array([self.height]))

        magik_dataframe[TIME_COLUMN_NAME] = magik_dataframe[TIME_COLUMN_NAME] * ((self.hyperparameters['number_of_frames_used_in_simulations'] - 1) * FRAME_RATE)

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

            v = [
                np.expand_dims(considered_nodes_features, 0),
                np.expand_dims(considered_edges_features, 0),
                np.expand_dims(considered_edges, 0),
                np.expand_dims(considered_edges_weights, 0),
            ]

            with get_device():
                predictions[initial_index:final_index] = (self.magik_architecture(v).numpy() > self.threshold)[0, ...] if apply_threshold else (self.magik_architecture(v).numpy())[0, ...]

        if not detect_clusters:
            return grapht[1][1], predictions

        """
        As the cluster detection is sensible to the edge pruning (if only one edge is misclassified as positive, two clusters are merged),
        detected connected components should be segmented to avoid this problem. If two clusters are merged and the performance of the
        edge classifier is too high, both clusters may be segmentated maximizing the modularity of graph partition. Hint: Levounian
        """

        edges_to_remove = np.where(predictions == 0)[0]
        remaining_edges_keep = np.delete(grapht[0][2], edges_to_remove, axis=0)

        #remaining_edges_weights = np.delete(grapht[0][1], edges_to_remove, axis=0) #Distance Weight
        #remaining_edges_weights = 1 / np.delete(grapht[0][1], edges_to_remove, axis=0) #Inverse Distance Weight

        G=nx.Graph()
        G.add_edges_from(remaining_edges_keep) #Unweight Graph
        #G.add_weighted_edges_from(np.hstack((remaining_edges_keep, remaining_edges_weights)))  #Weighted Graph

        cluster_sets = []

        """
        #Connected Components
        cluster_sets = nx.connected_components(G)
        """

        """
        #Louvain Method with Weights
        cluster_sets = nx.community.louvain_communities(G, weight='weight')
        """

        """
        #Louvain Method without Weights
        cluster_sets = nx.community.louvain_communities(G, weight=None)
        """

        """
        #Greedy Modularity with Weights
        cluster_sets = nx.community.greedy_modularity_communities(G, weight='weight')
        """

        #Greedy Modularity without Weights
        cluster_sets = nx.community.greedy_modularity_communities(G, weight=None)

        magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED] = 0

        for index, a_set in enumerate(cluster_sets):
            for value in a_set:
                magik_dataset.loc[value, MAGIK_LABEL_COLUMN_NAME_PREDICTED] = index + 1

        magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED] = magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED].astype(int)

        if MAGIK_LABEL_COLUMN_NAME in magik_dataset.columns:
            magik_dataset[MAGIK_LABEL_COLUMN_NAME] = magik_dataset[MAGIK_LABEL_COLUMN_NAME].astype(int)

        #Last Correction
        cluster_indexes_list = list(set(magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED]))

        if 0 in cluster_indexes_list:
            cluster_indexes_list.remove(0)

        for cluster_index in cluster_indexes_list:
            cluster_dataframe = magik_dataset[ magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED] == cluster_index ]

            if len(cluster_dataframe) >= 4 and cluster_dataframe[TIME_COLUMN_NAME].max() - cluster_dataframe[TIME_COLUMN_NAME].min() != 0:
                pass
            else:
                magik_dataset.loc[cluster_dataframe.index, CLUSTERIZED_COLUMN_NAME+'_predicted'] = 0
                magik_dataset.loc[cluster_dataframe.index, MAGIK_LABEL_COLUMN_NAME_PREDICTED] = 0

        #From here, there are correction. I need to measure how meaningful are these.
        """
        cluster_indexes_list = list(set(magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED]))
        cluster_indexes_list.remove(0)
        max_index = max(cluster_indexes_list)
        offset = 1

        for index in cluster_indexes_list:
            cluster_by_index = magik_dataset[[MAGIK_X_POSITION_COLUMN_NAME, MAGIK_Y_POSITION_COLUMN_NAME, TIME_COLUMN_NAME]][magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED] == index]
            points_index = cluster_by_index.index
            points = cluster_by_index.values
            stop = False
            n_clusters = 2
            picked_n_clusters = None

            while not stop:
                if len(points) == n_clusters:
                    picked_n_clusters = 1
                    break

                labels = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(points[:,0:2])

                alphashapes_list = []

                for label in set(labels):
                    cluster = points[:,0:2][labels == label]

                    if len(cluster) > 1:
                        alphashapes_list.append(alphashape.alphashape(cluster, 0))

                for a_i in range(0, len(alphashapes_list)):
                    for a_j in range(a_i+1, len(alphashapes_list)):
                        if alphashapes_list[a_i].intersects(alphashapes_list[a_j]):
                            stop = True
                            picked_n_clusters = n_clusters - 1
                            break

                    if stop:
                        break

                if stop:
                    break
                else:
                    n_clusters += 1
            
            labels = AgglomerativeClustering(n_clusters=picked_n_clusters, linkage='ward').fit_predict(points[:,0:2])
            #labels = MeanShift(n_jobs=-1).fit_predict(points[:,0:2]) #With Mean Shift
            magik_dataset.loc[points_index, MAGIK_LABEL_COLUMN_NAME_PREDICTED] = labels + offset + max_index
            offset += max(labels) + 1
        """

        """
        cluster_indexes_list = list(set(magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED]))
        cluster_indexes_list.remove(0)

        for cluster_index in cluster_indexes_list:
            cluster_dataframe = magik_dataset[magik_dataset['solution_predicted'] == cluster_index].copy()
            cluster_dataframe['index'] = cluster_dataframe.index

            if len(cluster_dataframe) >= 5:
                df_window = cluster_dataframe.copy()

                simplices = Delaunay(df_window[['position-x', 'position-y', 't']].values).simplices

                def less_first(a, b):
                    return [a,b] if a < b else [b,a]

                list_of_edges = []

                for triangle in simplices:
                    for e1, e2 in [[0,1],[1,2],[2,0]]: # for all edges of triangle
                        list_of_edges.append(less_first(triangle[e1],triangle[e2])) # always lesser index first

                new_index_to_old_index = {new_index:df_window.iloc[new_index]['index'] for new_index in range(len(df_window.index.values))}
                list_of_edges = np.vectorize(new_index_to_old_index.get)(list_of_edges)
                list_of_edges = np.unique(list_of_edges, axis=0).tolist() # remove duplicates
                list_of_edges_as_dataframe = pd.DataFrame({'index_x': [edge[0] for edge in list_of_edges], 'index_y': [edge[1] for edge in list_of_edges]})

                simplified_cross = list_of_edges_as_dataframe.merge(df_window.rename(columns={old_column_name: old_column_name+'_x' for old_column_name in df_window.columns}), on='index_x')
                simplified_cross = simplified_cross.merge(df_window.rename(columns={old_column_name: old_column_name+'_y' for old_column_name in df_window.columns}), on='index_y')

                df_window = simplified_cross.copy()
                df_window = df_window[df_window['index_x'] != df_window['index_y']]
                df_window['distance-x'] = df_window[f"{MAGIK_X_POSITION_COLUMN_NAME}_x"] - df_window[f"{MAGIK_X_POSITION_COLUMN_NAME}_y"]
                df_window['distance-y'] = df_window[f"{MAGIK_Y_POSITION_COLUMN_NAME}_x"] - df_window[f"{MAGIK_Y_POSITION_COLUMN_NAME}_y"]
                df_window['distance-t'] = df_window[f"t_x"] - df_window[f"t_y"]
                df_window['distance'] = ((df_window['distance-x']**2) + (df_window['distance-y']**2) + (df_window['distance-t']**2))**(1/2)
                df_window = df_window[df_window['distance'] < df_window['distance'].quantile(0.90)]

                G=nx.Graph()
                G.add_edges_from(df_window[['index_x', 'index_y']].values)
                cluster_sets = list(nx.connected_components(G))

                cluster_dataframe.loc[:,'solution_predicted'] = 0

                largest_sub_cluster = max(list(cluster_sets), key=len)

                for value in largest_sub_cluster:
                    cluster_dataframe.loc[value, 'solution_predicted'] = cluster_index

                magik_dataset.loc[cluster_dataframe['index'], 'solution_predicted'] = cluster_dataframe['solution_predicted'].astype(int)
        """

        """
        cluster_indexes_list = list(set(magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED]))
        cluster_indexes_list.remove(0)
        index_counter = 1

        for cluster_index in cluster_indexes_list:
            cluster_dataframe = magik_dataset[magik_dataset['solution_predicted'] == cluster_index].copy()
            cluster_dataframe['index'] = cluster_dataframe.index

            if len(cluster_dataframe) >= 5:
                df_window = cluster_dataframe.copy()

                simplices = Delaunay(df_window[['position-x', 'position-y', 't']].values).simplices

                def less_first(a, b):
                    return [a,b] if a < b else [b,a]

                list_of_edges = []

                for triangle in simplices:
                    for e1, e2 in [[0,1],[1,2],[2,0]]: # for all edges of triangle
                        list_of_edges.append(less_first(triangle[e1],triangle[e2])) # always lesser index first

                new_index_to_old_index = {new_index:df_window.iloc[new_index]['index'] for new_index in range(len(df_window.index.values))}
                list_of_edges = np.vectorize(new_index_to_old_index.get)(list_of_edges)
                list_of_edges = np.unique(list_of_edges, axis=0).tolist() # remove duplicates
                list_of_edges_as_dataframe = pd.DataFrame({'index_x': [edge[0] for edge in list_of_edges], 'index_y': [edge[1] for edge in list_of_edges]})

                simplified_cross = list_of_edges_as_dataframe.merge(df_window.rename(columns={old_column_name: old_column_name+'_x' for old_column_name in df_window.columns}), on='index_x')
                simplified_cross = simplified_cross.merge(df_window.rename(columns={old_column_name: old_column_name+'_y' for old_column_name in df_window.columns}), on='index_y')

                df_window = simplified_cross.copy()
                df_window = df_window[df_window['index_x'] != df_window['index_y']]
                df_window['distance-x'] = df_window[f"{MAGIK_X_POSITION_COLUMN_NAME}_x"] - df_window[f"{MAGIK_X_POSITION_COLUMN_NAME}_y"]
                df_window['distance-y'] = df_window[f"{MAGIK_Y_POSITION_COLUMN_NAME}_x"] - df_window[f"{MAGIK_Y_POSITION_COLUMN_NAME}_y"]
                df_window['distance-t'] = df_window[f"t_x"] - df_window[f"t_y"]
                df_window['distance'] = ((df_window['distance-x']**2) + (df_window['distance-y']**2))**(1/2)
                df_window = df_window[df_window['distance'] < df_window['distance'].quantile(0.75)]

                G=nx.Graph()
                G.add_edges_from(df_window[['index_x', 'index_y']].values)
                cluster_sets = list(nx.connected_components(G))

                cluster_dataframe.loc[:,'solution_predicted'] = 0

                for index, a_set in enumerate(cluster_sets):
                    for value in a_set:
                        cluster_dataframe.loc[value, 'solution_predicted'] = index_counter
                    index_counter += 1

                magik_dataset.loc[cluster_dataframe['index'], 'solution_predicted'] = cluster_dataframe['solution_predicted'].astype(int)
        """

        #Filtered Localization Reinsertion
        if original_dataset_path is not None:
            original_dataset = self.transform_smlm_dataset_to_magik_dataframe(pd.read_csv(original_dataset_path), ignored_non_clustered_localizations=False)

            for _, row in magik_dataset.iterrows():
                original_dataset.loc[original_dataset["original_index_for_recovery"] == row["original_index_for_recovery"], MAGIK_LABEL_COLUMN_NAME_PREDICTED] = row[MAGIK_LABEL_COLUMN_NAME_PREDICTED]
                original_dataset.loc[original_dataset["original_index_for_recovery"] == row["original_index_for_recovery"], CLUSTERIZED_COLUMN_NAME+'_predicted'] = row[CLUSTERIZED_COLUMN_NAME+'_predicted']

            magik_dataset = original_dataset

        """
        #Alpha-Shape Correction
        cluster_indexes_list = list(set(magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED]))
        cluster_indexes_list.remove(0)

        unclusterized_points = magik_dataset[[MAGIK_X_POSITION_COLUMN_NAME, MAGIK_Y_POSITION_COLUMN_NAME, TIME_COLUMN_NAME]][magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED] == 0].values
        unclusterized_points_index = magik_dataset.index.values
        unclusterized_info = [element for element in zip(unclusterized_points_index, unclusterized_points)]

        for cluster_index in cluster_indexes_list:
            magik_dataset_by_cluster_index = magik_dataset[magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED] == cluster_index]
            cluster_polygon = alphashape.alphashape(magik_dataset_by_cluster_index[[MAGIK_X_POSITION_COLUMN_NAME, MAGIK_Y_POSITION_COLUMN_NAME, TIME_COLUMN_NAME]].values, 0)

            for info in unclusterized_info:
                if cluster_polygon.contains(Point(*info[1])) and magik_dataset.loc[info[0], MAGIK_LABEL_COLUMN_NAME_PREDICTED] == 0:
                    magik_dataset.loc[info[0], MAGIK_LABEL_COLUMN_NAME_PREDICTED] = cluster_index
                    magik_dataset.loc[info[0], CLUSTERIZED_COLUMN_NAME+'_predicted'] = 1
        """

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
            list_of_edges_as_dataframe = pd.DataFrame({'index_x': [edge[0] for edge in list_of_edges], 'index_y': [edge[1] for edge in list_of_edges]})

            simplified_cross = list_of_edges_as_dataframe.merge(df_window.rename(columns={old_column_name: old_column_name+'_x' for old_column_name in df_window.columns}), on='index_x')
            simplified_cross = simplified_cross.merge(df_window.rename(columns={old_column_name: old_column_name+'_y' for old_column_name in df_window.columns}), on='index_y')

            df_window = simplified_cross.copy()
            df_window = df_window[df_window['index_x'] != df_window['index_y']]
            df_window['distance-x'] = df_window[f"{MAGIK_X_POSITION_COLUMN_NAME}_x"] - df_window[f"{MAGIK_X_POSITION_COLUMN_NAME}_y"]
            df_window['distance-y'] = df_window[f"{MAGIK_Y_POSITION_COLUMN_NAME}_x"] - df_window[f"{MAGIK_Y_POSITION_COLUMN_NAME}_y"]
            df_window['distance'] = ((df_window['distance-x']**2) + (df_window['distance-y']**2))**(1/2)

            if MAGIK_LABEL_COLUMN_NAME in full_nodes_dataset.columns:
                df_window['same_cluster'] = (df_window[MAGIK_LABEL_COLUMN_NAME+"_x"] == df_window[MAGIK_LABEL_COLUMN_NAME+"_y"])
            else:
                df_window['same_cluster'] = False

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
        return f"edge_classifier_batch_size_{self.hyperparameters['batch_size']}_partition_{self.hyperparameters['partition_size']}.h5"

    @property
    def predictions_file_name(self):
        return f"edge_classifier_batch_size_{self.hyperparameters['batch_size']}_partition_{self.hyperparameters['partition_size']}.csv"
    
    @property
    def history_training_info_file_name(self):
        return f"edge_classifier_batch_size_{self.hyperparameters['batch_size']}_partition_{self.hyperparameters['partition_size']}.json"

    @property
    def threshold_file_name(self):
        return f"edge_classifier_batch_size_{self.hyperparameters['batch_size']}_partition_{self.hyperparameters['partition_size']}.bin"

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

    def fit_with_datasets_from_path(self, path, save_checkpoints=False):
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
        with open(self.threshold_file_name, "w") as threshold_file:
            threshold_file.write(str(self.threshold))

    def save_keras_model(self):
        self.magik_architecture.save_weights(self.model_file_name)
        
    def load_keras_model(self):
        try:
            self.build_network()
            self.magik_architecture.load_weights(self.model_file_name)
        except FileNotFoundError:
            print(f"WARNING: {self} has not found keras model file (file name:{self.model_file_name})")
            return None

        return self.magik_architecture

    def load_threshold(self):
        try:
            with open(self.threshold_file_name, "r") as threshold_file:
                self.threshold = float(threshold_file.read())
        except FileNotFoundError:
            print(f"WARNING: {self} has not found keras model file (file name:{self.threshold_file_name})")
            return None

        return self.threshold

    def load_model(self):
        self.load_keras_model()
        self.load_threshold()
