"""
Divide an input graph into subgraphs
"""

import networkx as nx
import metis

def divide_graph(g, num):
  """
  Divide a large graph into subgraphs
  :param g: Input graph
  :param num: Number of partitions
  :return: Divided subgraphs without overlapping
  
  :type g: nx.Graph
  :type num: int
  :rtype: list
  """

  edge_cuts, parts = metis.part_graph(g, num, contig=True)
  
  member_map = dict(zip(g.nodes, parts))  # dictionary node ID -> part ID
  
  subgraphs = list()
  for i in range(num):
    members = [k for k, v in member_map.iteritems() if v == i]  # Extract members
    subgraph = g.subgraph(members)
    subgraphs.append(subgraph)
  
  return subgraphs


