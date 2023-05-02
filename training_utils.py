import numpy as np


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

        return (node_features, edge_features, edge_connections, weights), (
            node_labels,
            edge_labels,
            glob_labels,
        )

    return inner

def CustomGetSubGraph():
    def inner(data):
        graph, labels = data

        min_num_nodes = 2500
        max_num_nodes = 3000

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
