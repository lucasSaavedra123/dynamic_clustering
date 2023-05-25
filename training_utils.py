from collections import Counter

import numpy as np
import tensorflow as tf
import keras.backend as K

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