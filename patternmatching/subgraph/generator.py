"""
Generate subgraphs from an input graph
"""

import networkx as nx

from divide import divide_graph
from overlap import create_overlap


def generate_subgraphs(g, num, depth):
  """
  Generate subgraphs from an input graph
  :param g: Input graph
  :param num: Number of subgraphs
  :param depth: Overlapping depth
  :return:
  
  :type g: nx.Graph
  :type num: int
  :type depth: int
  :rtype: list
  """
  
  subgraphs = list()
  partitions = divide_graph(g, num)
  
  for part in partitions:
    subgraph = create_overlap(g, part, depth)
    subgraphs.append(subgraph)
  
  return subgraphs


