from math import sqrt
import os
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
from matplotlib.ticker import MaxNLocator
import keras.backend as K
from scipy.spatial import Delaunay
import more_itertools as mit
from operator import is_not
from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.spatial import ConvexHull
from matplotlib.ticker import FuncFormatter
import tqdm
import keras.backend as K
from scipy.spatial import Delaunay
import more_itertools as mit
import tqdm
from operator import is_not
from functools import partial

from .CONSTANTS import *


def save_number_in_file(file_name, number):
    with open(file_name, "w") as new_file:
        new_file.write(str(number))

def save_numbers_in_file(file_name, list_of_numbers):
    with open(file_name, "w") as new_file:
        for number in list_of_numbers:
            new_file.write(str(number)+'\n')

def save_number_lists_in_file(file_name, lists_of_numbers):
    with open(file_name, "w") as new_file:
        for list_of_numbers in lists_of_numbers:
            new_line = ''
            for number in list_of_numbers[:-1]:
                new_line += str(number)+','
            new_file.write(new_line+str(list_of_numbers[-1])+'\n')

def read_number_from_file(file_name, if_doesnt_exist_return=None):
    try:
        with open(file_name, "r") as file_to_read:
            return float(file_to_read.read())
    except FileNotFoundError:
        return if_doesnt_exist_return

def custom_norm(vector_one, vector_two):
  a = pow(vector_one[0] - vector_two[0], 2)
  b = pow(vector_one[1] - vector_two[1], 2)
  #assert np.linalg.norm(vector_one-vector_two) == sqrt(a+b)
  return sqrt(a+b)

def delete_file_if_exist(file_name):
    try:
        os.remove(file_name)
    except:
        pass

def positive_rate(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))

    return  (tp + fn) / (tp + tn + fp + fn)

def negative_rate(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))

    return  (fp + tn) / (tp + tn + fp + fn)

def CustomGetSubSet(ignore_non_cluster_experiments):
    def inner(data):
        graph, labels, sets = data

        retry = True

        while retry:
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

            counter = Counter(np.array(node_labels[:,0]))
            retry = ignore_non_cluster_experiments and (counter[0] == 0 or counter[1] == 0)

        return (node_features, edge_features, edge_connections, weights), (
            node_labels,
            edge_labels,
            glob_labels,
        )

    return inner

"""
def CustomGetSubSet():
    def inner(data):
        graph, labels, sets = data

        randset= np.random.randint(np.max(sets[0][:, 0]) + 1)

        nodeidxs = np.where(sets[0][:, 0] == randset)[0]
        edgeidxs = np.where(sets[1][:, 0] == randset)[0]

        node_features = graph[0][nodeidxs]
        edge_features = graph[1][edgeidxs]
        edge_connections = graph[2][edgeidxs]

        old_index_to_new_index = {old_index:new_index for new_index, old_index in enumerate(np.array(nodeidxs))}
        edge_connections = np.array(edge_connections)
        edge_connections = np.vectorize(old_index_to_new_index.get)(edge_connections)

        weights = graph[3][edgeidxs]

        node_labels = labels[0][nodeidxs]
        edge_labels = labels[1][edgeidxs]
        glob_labels = labels[2][randset]

        return (node_features, edge_features, edge_connections, weights), (
            node_labels,
            edge_labels,
            glob_labels,
        )

    return inner
"""

def CustomGetSubGraphByNumberOfNodes(min_num_nodes, max_num_nodes):
    def inner(data):
        graph, labels = data

        num_nodes = np.random.randint(min_num_nodes, max_num_nodes+1)
        node_start = np.random.randint(max(len(graph[0]) - num_nodes, 1))

        considered_nodes = list(range(node_start, node_start + num_nodes))
        considered_nodes_features = graph[0][considered_nodes]

        considered_edges_positions = np.all( np.isin(graph[2], considered_nodes), axis=-1)
        considered_edges_features = graph[1][considered_edges_positions]
        considered_edges = graph[2][considered_edges_positions]
        considered_edges_weights = graph[3][considered_edges_positions]
        
        old_index_to_new_index = {old_index:new_index for new_index, old_index in enumerate(considered_nodes)}
        considered_edges = np.array(considered_edges)
        considered_edges = np.vectorize(old_index_to_new_index.get)(considered_edges)
    
        node_labels = labels[0][considered_nodes]
        edge_labels = labels[1][considered_edges_positions]
        global_labels = labels[2]

        return (considered_nodes_features, considered_edges_features, considered_edges, considered_edges_weights), (
            node_labels,
            edge_labels,
            global_labels,
        )

    return inner

def CustomGetSubGraphByNumberOfEdges(min_num_edges, max_num_edges):
    def inner(data):
        graph, labels = data

        num_edges = np.random.randint(min_num_edges, max_num_edges+1)
        edge_start = np.random.randint(max(len(graph[1]) - num_edges, 1))

        considered_edges_features = graph[1][edge_start:edge_start+num_edges]
        considered_edges = graph[2][edge_start:edge_start+num_edges]
        considered_edges_weights = graph[3][edge_start:edge_start+num_edges]

        considered_nodes = np.unique(considered_edges.flatten())
        considered_nodes_features = graph[0][considered_nodes]

        old_index_to_new_index = {old_index:new_index for new_index, old_index in enumerate(considered_nodes)}
        considered_edges = np.vectorize(old_index_to_new_index.get)(considered_edges)
    
        node_labels = labels[0][considered_nodes]
        edge_labels = labels[1][edge_start:edge_start+num_edges]
        global_labels = labels[2]

        return (considered_nodes_features, considered_edges_features, considered_edges, considered_edges_weights), (
            node_labels,
            edge_labels,
            global_labels,
        )

    return inner

def CustomNodeBalancing():
    def inner(data):
        graph, labels = data

        boolean_array_if_node_is_clusterized = labels[0][:, 0] == 1
        boolean_array_if_node_is_not_clusterized = labels[0][:, 0] == 0

        number_of_clusterized_nodes = np.sum(np.array((boolean_array_if_node_is_clusterized)) * 1)
        number_of_non_clusterized_nodes = np.sum(np.array(boolean_array_if_node_is_not_clusterized) * 1)

        if number_of_clusterized_nodes != number_of_non_clusterized_nodes and number_of_non_clusterized_nodes != 0 and number_of_clusterized_nodes != 0:
            retry = True

            while retry:

                if number_of_clusterized_nodes > number_of_non_clusterized_nodes:
                    nodeidxs = np.array(np.where(boolean_array_if_node_is_clusterized)[0])
                    nodes_to_select = np.random.choice(nodeidxs, size=number_of_non_clusterized_nodes, replace=False)
                    nodes_to_select = np.append(nodes_to_select, np.array(np.where(boolean_array_if_node_is_not_clusterized)[0]))
                else:
                    nodeidxs = np.array(np.where(boolean_array_if_node_is_not_clusterized)[0])
                    nodes_to_select = np.random.choice(nodeidxs, size=number_of_clusterized_nodes, replace=False)
                    nodes_to_select = np.append(nodes_to_select, np.array(np.where(boolean_array_if_node_is_clusterized)[0]))

                nodes_to_select = sorted(nodes_to_select)

                id_to_new_id = {}

                for index, value in enumerate(nodes_to_select):
                    id_to_new_id[value] = index

                edge_connects_removed_node = np.any( ~np.isin(graph[2], nodes_to_select), axis=-1)

                node_features = graph[0][nodes_to_select]
                edge_features = graph[1][~edge_connects_removed_node]
                edge_connections = np.vectorize(id_to_new_id.get)(graph[2][~edge_connects_removed_node])
                weights = graph[3][~edge_connects_removed_node]

                node_labels = labels[0][nodes_to_select]
                edge_labels = labels[1][~edge_connects_removed_node]
                global_labels = labels[2]

                retry = node_features.shape[0] == edge_features.shape[0]

            return (node_features, edge_features, edge_connections, weights), (
                node_labels,
                edge_labels,
                global_labels,
            )
        else:
            return graph, labels

    return inner

def CustomAugmentCentroids(rotate, flip_x, flip_y):
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

def CustomEdgeBalancing():
    def inner(data):
        graph, labels = data

        boolean_array_if_node_is_same_cluster = labels[1][:, 0] == 1
        boolean_array_if_node_is_not_same_cluster = labels[1][:, 0] == 0

        number_of_same_cluster_edges = np.sum(np.array((boolean_array_if_node_is_same_cluster)) * 1)
        number_of_non_same_cluster_edges = np.sum(np.array(boolean_array_if_node_is_not_same_cluster) * 1)

        if number_of_same_cluster_edges != number_of_non_same_cluster_edges and number_of_same_cluster_edges != 0 and number_of_non_same_cluster_edges != 0:
            retry = True

            while retry:

                if number_of_same_cluster_edges > number_of_non_same_cluster_edges:
                    edgeidxs = np.array(np.where(boolean_array_if_node_is_same_cluster)[0])
                    edges_to_select = np.random.choice(edgeidxs, size=number_of_non_same_cluster_edges, replace=False)
                    edges_to_select = np.append(edges_to_select, np.array(np.where(boolean_array_if_node_is_not_same_cluster)[0]))
                else:
                    edgeidxs = np.array(np.where(boolean_array_if_node_is_not_same_cluster)[0])
                    edges_to_select = np.random.choice(edgeidxs, size=number_of_same_cluster_edges, replace=False)
                    edges_to_select = np.append(edges_to_select, np.array(np.where(boolean_array_if_node_is_same_cluster)[0]))                                

                nodes_to_select = sorted(np.unique(np.array(graph[2][edges_to_select])))

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

                retry = node_features.shape[0] == edge_features.shape[0]

            return (node_features, edge_features, edge_connections, weights), (
                node_labels,
                edge_labels,
                global_labels,
            )
        
        else:
            return graph, labels

    return inner

def get_device():
    return tf.device('/gpu:0' if len(tf.config.list_physical_devices('GPU')) == 1 else '/cpu:0')

def delaunay_from_dataframe(dataframe, columns_to_pick):
  list_of_edges = []

  simplices = Delaunay(dataframe[columns_to_pick].values).simplices

  def less_first(a, b):
      return [a,b] if a < b else [b,a]

  for simplex in simplices:
    if len(simplex) == 3:
        set_to_iterate = [[0,1],[0,2],[1,2]]
    elif len(simplex) == 4:
        set_to_iterate = [[0,1],[0,2],[1,2],[0,3],[1,3],[2,3]]
    else:
        raise Exception(f'Simplex of size {len(simplex)} are not allowed')

    for e1, e2 in set_to_iterate:
        list_of_edges.append(less_first(simplex[e1],simplex[e2]))

  return list_of_edges

def predict_on_dataset(smlm_dataset, localization_classifier, edge_classifier):
    TEMPORAL_FILE_NAME = 'for_delete.for_delete'

    magik_dataset = localization_classifier.transform_smlm_dataset_to_magik_dataframe(smlm_dataset)
    magik_dataset = localization_classifier.predict(magik_dataset, apply_threshold=True)
    smlm_dataset = localization_classifier.transform_magik_dataframe_to_smlm_dataset(magik_dataset)

    smlm_dataset.to_csv(TEMPORAL_FILE_NAME)

    magik_dataset = edge_classifier.transform_smlm_dataset_to_magik_dataframe(smlm_dataset)
    magik_dataset = edge_classifier.predict(magik_dataset, detect_clusters=True, apply_threshold=True, original_dataset_path=TEMPORAL_FILE_NAME)
    smlm_dataset = edge_classifier.transform_magik_dataframe_to_smlm_dataset(magik_dataset)

    os.remove(TEMPORAL_FILE_NAME)

    return smlm_dataset

def build_graph_with_spatio_temporal_criterion(full_nodes_dataset, radius, nofframes, edge_features, node_features, return_real_edges_weights=False):
        edges_dataframe = pd.DataFrame({
            "distance": [],
            "index_1": [],
            "index_2": [],
            MAGIK_DATASET_COLUMN_NAME: [],
        })

        full_nodes_dataset = full_nodes_dataset.copy()

        sets = np.unique(full_nodes_dataset[MAGIK_DATASET_COLUMN_NAME])

        for setid in sets:
            df_set = full_nodes_dataset[full_nodes_dataset[MAGIK_DATASET_COLUMN_NAME] == setid].copy().reset_index()

            maxframe = range(0, df_set[FRAME_COLUMN_NAME].max() + 1 + nofframes)

            windows = mit.windowed(maxframe, n=nofframes, step=1)
            windows = map(lambda x: list(filter(partial(is_not, None), x)), windows)
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
                df_window['t-difference'] = df_window[f"{TIME_COLUMN_NAME}_x"] - df_window[f"{TIME_COLUMN_NAME}_y"]
                df_window['real_distance'] = ((df_window['distance-x']**2) + (df_window['distance-y']**2) + (df_window['t-difference']**2))**(1/2)
                df_window = df_window[df_window['distance'] < radius]

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
                    'real_distance': [value[0] for value in df_window[["real_distance"]].values.tolist()],
                    'same_cluster': [value[0] for value in df_window[["same_cluster"]].values.tolist()]
                })], ignore_index=True)

            new_edges_dataframe = new_edges_dataframe.drop_duplicates()
            new_edges_dataframe[MAGIK_DATASET_COLUMN_NAME] = setid
            edges_dataframe = pd.concat([edges_dataframe, new_edges_dataframe], ignore_index=True)

        edgefeatures = edges_dataframe[edge_features].to_numpy()
        sparseadjmtx = edges_dataframe[["index_1", "index_2"]].to_numpy().astype(int)
        nodefeatures = full_nodes_dataset[node_features].to_numpy()

        edgeweights = np.ones(sparseadjmtx.shape[0])
        edgeweights = np.stack((np.arange(0, edgeweights.shape[0]), edgeweights), axis=1)

        nfsolution = full_nodes_dataset[[MAGIK_LABEL_COLUMN_NAME]].to_numpy()
        efsolution = edges_dataframe[['same_cluster']].to_numpy().astype(int)

        nodesets = full_nodes_dataset[[MAGIK_DATASET_COLUMN_NAME]].to_numpy().astype(int)
        edgesets = edges_dataframe[[MAGIK_DATASET_COLUMN_NAME]].to_numpy().astype(int)
        framesets = full_nodes_dataset[[FRAME_COLUMN_NAME]].to_numpy().astype(int)

        global_property = np.zeros(np.unique(full_nodes_dataset[MAGIK_DATASET_COLUMN_NAME]).shape[0])

        grapht = (
            (nodefeatures, edgefeatures, sparseadjmtx, edgeweights),
            (nfsolution, efsolution, global_property),
            (nodesets, edgesets, framesets)
        )

        if return_real_edges_weights:
            return grapht, edges_dataframe['real_distance'].to_numpy()
        else:
            return grapht

def styled_plotting(
        dataset_path,
        t_limit=None,
        x_limit=None,
        y_limit=None,
        detection_alpha=0.1,
        with_clustering=False,
        spatial_unit='um',
        save_plot=False,
        plot_trajectories=False
    ):
    """
    This function recreates localization datasets plotting from the following paper:

    Super-resolved trajectory-derived nanoclustering analysis using spatiotemporal indexing
    T. P. Wallis, A. Jiang, K. Young, H. Hou, K. Kudo, A. J. McCann, et al.
    Nature Communications 2023 Vol. 14 Issue 1 Pages 3353
    DOI: 10.1038/s41467-023-38866-y
    """

    dataset = pd.read_csv(dataset_path)
    dataset = dataset.groupby(['x', 'y']).first().reset_index()

    if t_limit is not None:
        assert t_limit[0] <= t_limit[1]
        dataset = dataset[dataset['t'] >= t_limit[0]].copy()
        dataset = dataset[dataset['t'] <= t_limit[1]].copy()

    dataset['normalized_time'] = dataset['t'] - dataset['t'].min()
    dataset['normalized_time'] /= dataset['normalized_time'].max()

    if x_limit is not None:
        assert x_limit[0] <= x_limit[1]
        dataset = dataset[dataset['x'] >= x_limit[0]].copy()
        dataset = dataset[dataset['x'] <= x_limit[1]].copy()

    if y_limit is not None:
        assert y_limit[0] <= y_limit[1]
        dataset = dataset[dataset['y'] >= y_limit[0]].copy()
        dataset = dataset[dataset['y'] <= y_limit[1]].copy()

    if CLUSTER_ID_COLUMN_NAME+'_predicted' in dataset.columns and with_clustering:
        cluster_ids = np.unique(dataset[CLUSTER_ID_COLUMN_NAME+'_predicted'].values).tolist()
        cluster_ids.remove(0)
    else:
        cluster_ids = []

    plt.rcdefaults() 
    font = {"family" : "Arial","size": 12} 
    matplotlib.rc('font', **font)

    ax0 = plt.subplot(111)
    ax0.cla()

    x_plot,y_plot,t_plot= dataset['x'].values.tolist(), dataset['y'].values.tolist(), dataset['t'].values.tolist()

    ax0.scatter(x_plot,y_plot,c="w",s=3,linewidth=0,alpha=detection_alpha)

    ax0.set_facecolor("k")
    ax0.set_xlabel(f"X [{spatial_unit}]")
    ax0.set_ylabel(f"Y [{spatial_unit}]")

    xlims = plt.xlim() if x_limit is None else plt.xlim(x_limit)
    ylims = plt.ylim() if y_limit is None else plt.ylim(y_limit)

    cmap = matplotlib.cm.get_cmap('brg')
    ax0.imshow([[0,1], [0,1]], extent = (xlims[0],xlims[1],ylims[0],ylims[1]), cmap = cmap, interpolation = 'bicubic', alpha=0)
    plt.tight_layout()

    if plot_trajectories:
        for particle_id in np.unique(dataset[PARTICLE_ID_COLUMN_NAME].values).tolist():
            particle_data = dataset[dataset[PARTICLE_ID_COLUMN_NAME] == particle_id].sort_values('t')[['x', 'y']]
            ax0.add_artist(matplotlib.lines.Line2D(particle_data[X_POSITION_COLUMN_NAME],particle_data[Y_POSITION_COLUMN_NAME],c='w',alpha=0.25,linewidth=0.5)) 

    if with_clustering:
        cmap = matplotlib.cm.get_cmap('brg')
        
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])

        for cluster_id in cluster_ids:
            cluster_data = dataset[dataset[CLUSTER_ID_COLUMN_NAME+'_predicted'] == cluster_id][['x', 'y', 'normalized_time']].copy()
            cluster_data_positions = cluster_data[['x', 'y']].values

            if len(cluster_data_positions) >= 3:
                hull = ConvexHull(cluster_data_positions)

                for simplex in hull.simplices:
                    ax0.plot(cluster_data_positions[simplex, 0], cluster_data_positions[simplex, 1], c=cmap(cluster_data['normalized_time'].mean()))

        cbaxes = inset_axes(ax0, width="30%", height="3%", loc=3)
        fmt = lambda x, pos: r'     $t_{min}$' if pos == 0 else r'$t_{max}$'
        cbar = plt.colorbar(sm, cax=cbaxes, orientation='horizontal', ticks=[0, 1], ticklocation='top', format=FuncFormatter(fmt))
        cbar.ax.xaxis.set_tick_params(pad=0, labelsize=15)
        
        #cbar.set_label('Time', color='white', fontsize=10, labelpad=-7)
        plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='white')

    ax0.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax0.yaxis.set_major_locator(MaxNLocator(integer=True))

    if save_plot:
        if x_limit is not None or y_limit is not None:
            plt.savefig(dataset_path+'_styled_figure_sub_roi.png',dpi=700, bbox_inches='tight', pad_inches=0)
        else:
            plt.savefig(dataset_path+'_styled_figure.png',dpi=700, bbox_inches='tight', pad_inches=0)
            
    else:
        plt.show()
