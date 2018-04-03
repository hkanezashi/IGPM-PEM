"""
Patrial Execution Manager
- Renforcement learning component to compute parameters for clustering
- Graph clustering by Louvain method
"""

import networkx as nx
import community  # http://python-louvain.readthedocs.io/en/latest/


class PEM:
  
  def __init__(self, g):
    """
    :type g: nx.Graph
    :param g: Input data graph
    """
    self.g = g
    
  
  def get_recompute_nodes(self, affected, resolution):
    """
    Get nodes for recomputations
    :param affected: Affected nodes for graph updates
    :param resolution: Will change the size of the communities, represents the time described in
        "Laplacian Dynamics and Multiscale Modular Structure in Networks", R. Lambiotte, J.-C. Delvenne, M. Barahona
        https://arxiv.org/pdf/0812.1770.pdf
    :return: Set of nodes for recomputations
    """
    part = community.best_partition(self.g)  # node, partID
    
    nodes = set()
    
    return nodes
    
    


