import os
import logging
logging.disable(logging.WARNING)

import pandas as pd
import numpy as np
import tensorflow as tf
import deeptrack as dt
from deeptrack.models.gnns.generators import GraphGenerator, GraphExtractor

from CONSTANTS import *

class SubClusterLinker():
    def __init__(self, height=10, width=10, radius=0.2, nofframes=10):
        self._output_type = "edges"

        self.magik_variables = dt.DummyFeature(
            radius=radius,
            output_type=self._output_type,
            nofframes=nofframes, # time window to associate nodes (in frames) 
        )

        self.height = height
        self.width = width

        self.magik_architecture = None

    def build_network(self):
        self.magik_architecture = dt.models.gnns.MAGIK(
            dense_layer_dimensions=(64, 96,),       # number of features in each dense encoder layer
            base_layer_dimensions=(96, 96, 96),     # Latent dimension throughout the message passing layers
            number_of_node_features=2,              # Number of node features in the graphs
            number_of_edge_features=1,              # Number of edge features in the graphs
            number_of_edge_outputs=1,               # Number of predicted features
            node_output_activation="sigmoid",       # Activation function for the output layer
            output_type=self._output_type,                    # Output type. Either "edges", "nodes", or "graph"
        )

        self.magik_architecture.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss = 'binary_crossentropy',
            metrics=['accuracy'],
        )          

        self.magik_architecture.summary()

    def transform_smlm_dataset_to_magik_dataframe(self, smlm_dataframe, set_number = 0):
        smlm_dataframe = smlm_dataframe.rename(columns={
            X_POSITION_COLUMN_NAME: "centroid-0",
            Y_POSITION_COLUMN_NAME: "centroid-1",
            CLUSTERIZED_COLUMN_NAME: "solution",
            PARTICLE_ID_COLUMN_NAME: "label",
        })

        smlm_dataframe = smlm_dataframe.drop(TIME_COLUMN_NAME, axis=1)
        smlm_dataframe = smlm_dataframe.drop(CLUSTER_ID_COLUMN_NAME, axis=1)

        # normalize centroids between 0 and 1
        smlm_dataframe.loc[:, smlm_dataframe.columns.str.contains("centroid")] = (
            smlm_dataframe.loc[:, smlm_dataframe.columns.str.contains("centroid")]
            / np.array([self.width, self.height])
        )

        smlm_dataframe['set'] = set_number
        smlm_dataframe['solution'] = smlm_dataframe['solution'].astype(float)
        smlm_dataframe['label'] = smlm_dataframe['label'] - (min(smlm_dataframe['label']) - 1)

        return smlm_dataframe

    def get_dataset_from_path(self, path, set_number = 0):
        return self.transform_smlm_dataset_to_magik_dataframe(pd.read_csv(path), set_number=set_number)

    def get_datasets_from_path(self, path):
        file_names = [file_name for file_name in os.listdir(path) if file_name.endswith(".csv")]

        full_dataset = pd.DataFrame({})

        for csv_file_index, csv_file_name in enumerate(file_names):
            full_dataset = full_dataset.append(self.get_dataset_from_path(os.path.join(path, csv_file_name), set_number=csv_file_index))

        return full_dataset

    def get_magik_graph_from_transformed_smlm_dataset(self, transformed_smlm_dataset):
        return GraphExtractor(
            nodesdf=transformed_smlm_dataset, properties=["centroid"], validation=True, **self.magik_variables.properties()
        )

    def predict(self, transformed_smlm_dataset):
        grapht = self.get_magik_graph_from_transformed_smlm_dataset(transformed_smlm_dataset)

        v = [
            np.expand_dims(grapht[0][0][:, 1:], 0),
            np.expand_dims(grapht[0][1], 0),
            np.expand_dims(grapht[0][2][:, 2:], 0),
            np.expand_dims(grapht[0][3], 0),
        ]

        output_node_f = self.magik_architecture(v).numpy()
        pred = (output_node_f > 0.5)[0, ...]
        g = grapht[1][1]

        return pred, g, output_node_f, grapht

    def fit_with_datasets_from_path(self, path):
        full_nodes_dataset = self.get_datasets_from_path(path)
        self.build_network()

        def GetFeature(full_graph, **kwargs):
            return (
              dt.Value(full_graph)
              #>> dt.Lambda(
              #    AugmentCentroids,
              #    rotate=lambda: np.random.rand() * 2 * np.pi,
              #    #translate=lambda: np.random.randn(2) * 0.05,
              #    translate=lambda: np.random.randn(2) * 0,
              #    flip_x=lambda: np.random.randint(2),
              #    flip_y=lambda: np.random.randint(2),
              #  )
            )

        generator = GraphGenerator(
            nodesdf=full_nodes_dataset,
            properties=["centroid"],
            feature_function=GetFeature,
            min_data_size=5,
            max_data_size=11,
            batch_size=1,
            **self.magik_variables.properties()
        )

        with generator:
            self.magik_architecture.fit(generator, epochs=10)
