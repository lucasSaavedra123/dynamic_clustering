import os
import tqdm
import pickle
import json
from collections import Counter

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import deeptrack as dt
import numpy as np
import pandas as pd
import tqdm
import ghostml
from deeptrack.models.gnns.generators import ContinuousGraphGenerator

from CONSTANTS import *
from utils import *


class LocalizationClassifier():
    @classmethod
    def default_hyperparameters(cls):
        return {
            "partition_size": 3500,
            "epochs": 100,
            "batch_size": 1,
            "training_set_in_epoch_size": 512,
            "number_of_frames_used_in_simulations": 1000,
            "ignore_no_clusters_experiments_during_training": True
        }

    @classmethod
    def analysis_hyperparameters(cls):
        return {
            "partition_size": [500,1000,1500,2000,2500,3000,3500,4000]
        }

    def __init__(self, height=10, width=10, static=False):
        self._output_type = "nodes"

        self.magik_architecture = None
        self.history_training_info = None
        self.threshold = 0.5

        self.hyperparameters = self.__class__.default_hyperparameters()
        self.height = height
        self.width = width
        self.static = static

    @property
    def node_features(self):
        return [MAGIK_X_POSITION_COLUMN_NAME, MAGIK_Y_POSITION_COLUMN_NAME]

    @property
    def edge_features(self):
        return ["distance"]

    def set_dimensions(self, height, width):
        self.height = height
        self.width = width

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
            optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=['accuracy', tf.keras.metrics.AUC(), positive_rate, negative_rate]
        )

        #self.magik_architecture.summary()

    def transform_smlm_dataset_to_magik_dataframe(self, smlm_dataframe, set_number=0):
        smlm_dataframe = smlm_dataframe.rename(columns={
            X_POSITION_COLUMN_NAME: MAGIK_X_POSITION_COLUMN_NAME,
            Y_POSITION_COLUMN_NAME: MAGIK_Y_POSITION_COLUMN_NAME,
            CLUSTERIZED_COLUMN_NAME: MAGIK_LABEL_COLUMN_NAME,
            CLUSTERIZED_COLUMN_NAME+"_predicted": MAGIK_LABEL_COLUMN_NAME_PREDICTED
        })

        smlm_dataframe = smlm_dataframe.drop(["Unnamed: 0"], axis=1, errors="ignore")
        smlm_dataframe.loc[:, smlm_dataframe.columns.str.contains(MAGIK_POSITION_COLUMN_NAME)] = (smlm_dataframe.loc[:, smlm_dataframe.columns.str.contains(MAGIK_POSITION_COLUMN_NAME)] / np.array([self.width, self.height]))

        if TIME_COLUMN_NAME in smlm_dataframe.columns:
            smlm_dataframe[TIME_COLUMN_NAME] = smlm_dataframe[TIME_COLUMN_NAME] / ((self.hyperparameters['number_of_frames_used_in_simulations'] - 1) * FRAME_RATE)
        else:
            smlm_dataframe[TIME_COLUMN_NAME] = smlm_dataframe[FRAME_COLUMN_NAME] / (self.hyperparameters['number_of_frames_used_in_simulations'] - 1)
            smlm_dataframe['DUMMY'] = 'to_remove'

        smlm_dataframe[MAGIK_DATASET_COLUMN_NAME] = set_number

        if MAGIK_LABEL_COLUMN_NAME in smlm_dataframe.columns:
            smlm_dataframe[MAGIK_LABEL_COLUMN_NAME] = smlm_dataframe[MAGIK_LABEL_COLUMN_NAME].astype(int)
        else:
            smlm_dataframe[MAGIK_LABEL_COLUMN_NAME] = 0

        smlm_dataframe[MAGIK_LABEL_COLUMN_NAME_PREDICTED] = 0

        if not self.static:
            smlm_dataframe = smlm_dataframe.sort_values(TIME_COLUMN_NAME, ascending=True, inplace=False)

        return smlm_dataframe.reset_index(drop=True)

    def transform_magik_dataframe_to_smlm_dataset(self, magik_dataframe):
        magik_dataframe.loc[:, magik_dataframe.columns.str.contains(MAGIK_POSITION_COLUMN_NAME)] = (magik_dataframe.loc[:, magik_dataframe.columns.str.contains(MAGIK_POSITION_COLUMN_NAME)] * np.array([self.width, self.height]))

        if not self.static:
            if 'DUMMY' in magik_dataframe.columns:
                magik_dataframe = magik_dataframe.drop(['DUMMY', TIME_COLUMN_NAME], axis=1, errors="ignore")
            else:
                magik_dataframe[TIME_COLUMN_NAME] = magik_dataframe[TIME_COLUMN_NAME] * ((self.hyperparameters['number_of_frames_used_in_simulations'] - 1) * FRAME_RATE)

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
        if not self.static:
            return [os.path.join(path, file_name) for file_name in os.listdir(path) if file_name.endswith(".csv") and len(file_name.split('.'))==2]
        else:
            return [os.path.join(path, file_name) for file_name in os.listdir(path) if file_name.endswith(".tsv.csv")]

    def get_datasets_from_path(self, path):
        full_dataset = pd.DataFrame({})

        for csv_file_index, csv_file_path in enumerate(self.get_dataset_file_paths_from(path)):
            full_dataset = pd.concat([full_dataset, self.get_dataset_from_path(csv_file_path, set_number=csv_file_index)], ignore_index=True)

        return full_dataset.reset_index(drop=True)

    def predict(self, magik_dataset, apply_threshold=True, verbose=True):
        magik_dataset = magik_dataset.copy()

        grapht = self.build_graph(magik_dataset, verbose=False)

        predictions = np.empty((len(grapht[0][0]), 1))

        number_of_nodes = len(grapht[0][0])
        partitions_initial_index = list(range(0,number_of_nodes,self.hyperparameters['partition_size']))

        for index, initial_index in enumerate(partitions_initial_index):
            if index == len(partitions_initial_index)-1:
                final_index = number_of_nodes
            else:
                final_index = partitions_initial_index[index+1]

            considered_nodes = list(range(initial_index, final_index))
            considered_nodes_features = grapht[0][0][considered_nodes]

            considered_edges_positions = np.all( np.isin(grapht[0][2], considered_nodes), axis=-1)
            considered_edges_features = grapht[0][1][considered_edges_positions]
            considered_edges = grapht[0][2][considered_edges_positions]
            considered_edges_weights = grapht[0][3][considered_edges_positions]

            if considered_edges.size != 0:
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

        #magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED] = grapht[1][0] #Perfect Classification
        magik_dataset[MAGIK_LABEL_COLUMN_NAME_PREDICTED] = predictions

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

        iterator = tqdm.tqdm(sets) if verbose else sets

        for setid in iterator:
            new_edges_dataframe = pd.DataFrame({'index_1': [], 'index_2': [],'distance': []})
            df_window = full_nodes_dataset[full_nodes_dataset[MAGIK_DATASET_COLUMN_NAME] == setid].copy().reset_index()

            if self.static:
                list_of_edges = delaunay_from_dataframe(df_window, [MAGIK_X_POSITION_COLUMN_NAME, MAGIK_Y_POSITION_COLUMN_NAME])
            else:
                list_of_edges = delaunay_from_dataframe(df_window, [MAGIK_X_POSITION_COLUMN_NAME, MAGIK_Y_POSITION_COLUMN_NAME, TIME_COLUMN_NAME])

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
            
            edges = [sorted(edge) for edge in df_window[["index_x", "index_y"]].values.tolist()]

            new_edges_dataframe = pd.concat([new_edges_dataframe, pd.DataFrame({
                'index_1': [edge[0] for edge in edges],
                'index_2': [edge[1] for edge in edges],
                'distance': [value[0] for value in df_window[["distance"]].values.tolist()],
            })], ignore_index=True)

            new_edges_dataframe = new_edges_dataframe.drop_duplicates()
            new_edges_dataframe['set'] = setid
            
            edges_dataframe = pd.concat([edges_dataframe, new_edges_dataframe], ignore_index=True)

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
        return f"node_classifier.tmp"

    @property
    def model_file_name(self):
        return f"node_classifier_batch_size_{self.hyperparameters['batch_size']}_partition_{self.hyperparameters['partition_size']}.h5"

    @property
    def history_training_info_file_name(self):
        return f"node_classifier_batch_size_{self.hyperparameters['batch_size']}_partition_{self.hyperparameters['partition_size']}.json"

    @property
    def predictions_file_name(self):
        return f"node_classifier_batch_size_{self.hyperparameters['batch_size']}_partition_{self.hyperparameters['partition_size']}.csv"

    @property
    def threshold_file_name(self):
        return f"node_classifier_batch_size_{self.hyperparameters['batch_size']}_partition_{self.hyperparameters['partition_size']}.bin"

    def test_with_datasets_from_path(self, path, plot=False, apply_threshold=True, save_result=False, save_predictions=False, verbose=True, check_if_predictions_file_name_exists=False):
        if check_if_predictions_file_name_exists and os.path.exists(self.predictions_file_name):
            dataframe = pd.read_csv(self.predictions_file_name)
            true, pred = dataframe['true'].values.tolist(), dataframe['pred'].values.tolist()
        else:
            true = []
            pred = []

            iterator = tqdm.tqdm(self.get_dataset_file_paths_from(path)) if verbose else self.get_dataset_file_paths_from(path)

            for csv_file_name in iterator:
                dataset = self.get_dataset_from_path(csv_file_name)

                if not self.hyperparameters["ignore_no_clusters_experiments_during_training"] or not (not apply_threshold and len(dataset[dataset['solution'] == 1]) == 0):
                    r = self.predict(dataset, apply_threshold=apply_threshold, verbose=False)

                    if save_predictions:
                        r.to_csv(csv_file_name+f"_predicted_with_batch_size_{self.hyperparameters['batch_size']}_partition_{self.hyperparameters['partition_size']}.csv", index=False)

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

    def fit_with_datasets_from_path(self, path, save_checkpoints=False):

        if self.load_keras_model() is None:
            if os.path.exists(self.train_full_graph_file_name):
                fileObj = open(self.train_full_graph_file_name, 'rb')
                train_full_graph = pickle.load(fileObj)
                fileObj.close()
            else:
                train_full_graph = self.build_graph(self.get_datasets_from_path(path))
                fileObj = open(self.train_full_graph_file_name, 'wb')
                pickle.dump(train_full_graph, fileObj)
                fileObj.close()

            def CustomGetFeature(full_graph, **kwargs):
                return (
                    dt.Value(full_graph)
                    >> dt.Lambda(CustomGetSubSet,
                        ignore_non_cluster_experiments=lambda: self.hyperparameters["ignore_no_clusters_experiments_during_training"]
                    )
                    >> dt.Lambda(CustomGetSubGraphByNumberOfNodes,
                        min_num_nodes=lambda: self.hyperparameters["partition_size"],
                        max_num_nodes=lambda: self.hyperparameters["partition_size"]
                    )
                    #>> dt.Lambda(CustomNodeBalancing)
                    >> dt.Lambda(
                        CustomAugmentCentroids,
                        rotate=lambda: np.random.rand() * 2 * np.pi,
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
            true = []
            pred = []

            true, pred = self.test_with_datasets_from_path(path, apply_threshold=False, save_result=False, verbose=True)

            count = Counter(true)
            positive_is_majority = count[1] > count[0]

            if positive_is_majority:
                true = 1 - np.array(true)
                pred = 1 - np.array(pred)

            thresholds = np.round(np.arange(0.05,0.95,0.025), 3)

            if self.static:
                self.threshold = ghostml.optimize_threshold_from_predictions(true, pred, thresholds, ThOpt_metrics = 'ROC', N_subsets=1, subsets_size=0.01, with_replacement=False)
            else:
                self.threshold = ghostml.optimize_threshold_from_predictions(true, pred, thresholds, ThOpt_metrics = 'ROC', N_subsets=100, subsets_size=0.2, with_replacement=False)

            if positive_is_majority:
                self.threshold = 1 - self.threshold

            if save_checkpoints:
                self.save_threshold()

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

    def save_history_training_info(self):
        with open(self.history_training_info_file_name, "w") as json_file:
            json.dump(self.history_training_info, json_file)

    def save_threshold(self):
        save_number_in_file(self.threshold_file_name, self.threshold)

    def save_keras_model(self):
        self.magik_architecture.save_weights(self.model_file_name)

    def save_model(self):
        self.save_keras_model()
        self.save_threshold()

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
