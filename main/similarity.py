import numpy as np
import networkx as nx

def intra_layer_similarity(G):
    n = len(G.nodes)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            paths = list(nx.all_shortest_paths(G, source=i, target=j, weight='weight'))
            similarity_matrix[i][j] = similarity_matrix[j][i] = len(paths)
    return similarity_matrix

def inter_layer_similarity(G1, G2):
    similarity = 0
    for node in G1.nodes:
        if node in G2:
            deg1 = G1.degree(node, weight='weight')
            deg2 = G2.degree(node, weight='weight')
            similarity += deg1 * deg2
    return similarity
