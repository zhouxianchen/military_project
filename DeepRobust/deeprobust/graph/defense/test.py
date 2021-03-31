import torch
import numpy as np
import dgl
from dgl import DGLGraph
import dgl.function as fn
import networkx as nx
import scipy.sparse as sp




G =DGLGraph()
G.add_nodes(3)
G.add_edges([0, 1], 2)

# Set the function defined to be the default message function.


#
# def random_adj():
#     aa = np.random.rand(10,10)
#     return aa
# #
# # print(random_adj())
#
#
# graph = nx.from_numpy_matrix(random_adj(),create_using=nx.MultiGraph)
# print(graph[0][0])
#
#
#
# g = DGLGraph()
# g.from_networkx(graph, edge_attrs=['weight'])
# print(g.edata['weight'])
