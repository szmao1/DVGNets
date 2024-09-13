import networkx as nx
import numpy as np

def price_to_visibility_graph(carbon_prices):
    G = nx.Graph()
    n = len(carbon_prices)
    
    for i in range(n):
        G.add_node(i, value=carbon_prices[i])
        
    for i in range(n):
        for j in range(i + 1, n):
            visible = True
            for k in range(i + 1, j):
                if carbon_prices[k] >= carbon_prices[j] + (carbon_prices[i] - carbon_prices[j]) * (j - k) / (j - i):
                    visible = False
                    break
            if visible:
                G.add_edge(i, j, weight=abs(carbon_prices[i] - carbon_prices[j]))

    return G

def add_weights_to_graph(G, carbon_prices):
    for i, j in G.edges():
        dist = abs(j - i)
        weight = abs(carbon_prices[i] - carbon_prices[j]) / (dist ** 2)
        G[i][j]['weight'] = weight
    return G

def create_multiplex_visibility_networks(external_factors):
    multiplex_networks = []
    for factor_data in external_factors:
        G = price_to_visibility_graph(factor_data)
        G_weighted = add_weights_to_graph(G, factor_data)
        multiplex_networks.append(G_weighted)
    return multiplex_networks
