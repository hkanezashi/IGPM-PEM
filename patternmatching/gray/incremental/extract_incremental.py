"""
EXTRACT algorithm implementation

Tong, Hanghang, and Christos Faloutsos. "Center-piece subgraphs: problem definition and fast solutions."
Proceedings of the 12th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2006.
"""


from patternmatching.query.Condition import *
from patternmatching.gray.rwr import RWR_WCC

MAX_LENGTH = 5

class Extract:
  
  def __init__(self, g, rwr, label=None):
    """
    :type rwr: RWR_WCC
    """
    self.pre = dict()
    self.rwr = rwr
    self.g = g
    self.label = label
    self.default_value = 1.0 / g.number_of_nodes()
  
  def getRWR(self, i, j):
    # if i not in self.rwr:
    #   return self.default_value
    # else:
    #   return self.rwr[i].get(j, self.default_value)
    # return self.rwr[i][j]
    # v = self.rwr.get_value(i, j)
    # if v == 0.0:
    #   return self.default_value
    # else:
    #   return v
    return self.rwr.get_value(i, j)
  
  def computeExtract_batch(self):
    """Compute all neighbors and paths
    
    :return:
    """
    for i in self.g.nodes():
      self.pre[i] = dict()
      self.pre[i][i] = i
      self.computeExtractSingle(i)
  
  def computeExtract_incremental(self, nodes):
    """Compute neighbors and paths for specified node set
    
    :param nodes: Node set for recomputations
    :return:
    """
    for n in nodes:
      self.computeExtractSingle(n)
      

  def computeExtractSingle(self, i):
    """Compute from the specified single node
    
    :param i: Start node ID
    :return:
    """
    # print("ComputeExtractSingle: " + str(i))
    if i not in self.pre:
      self.pre[i] = dict()
      self.pre[i][i] = i
    
    dist = dict()   ## Distance score
    hops = dict()   ## Hops
    finished = set()   ## Finished set
    V = {i}
    dist[i] = self.getRWR(i, i)
    hops[i] = 1
    for u in V:
      if i != u:
        dist[u] = 0
        hops[u] = 0
    
    while V:
      max_d = 0.0
      u = None # V[0]
      for u_ in V:
        if dist[u_] > max_d:
          max_d = dist[u_]
          u = u_
      if u is None:
        return
      V.remove(u)
      finished.add(u)
      
      if u in hops:
        if hops[u] >= MAX_LENGTH:
          continue
      else:
        hops[u] = 0
      
      for v in self.g.neighbors(u):
        if self.label is not None and not self.label in Condition.get_edge_labels(self.g, u, v).values():
          continue
        if not v in finished:
          V.add(v)
        rw = self.getRWR(i, v)
        lu = hops[u]
        d = (rw + dist[u] * lu)/(lu + 1)
        if (not v in dist) or (dist[v] < d):
          dist[v] = d
          hops[v] = lu + 1
          self.pre[i][v] = u
  
  ## Extract the best path i -> j
  def getPath(self, i, j):
    lst = list()
    if not i in self.pre:
      return lst
    if not j in self.pre[i]:
      return lst
    v = j
    while v != i:
      lst.append(v)
      if not v in self.pre[i]:
        return []
      v = self.pre[i][v]
    lst.reverse()
    # print i, j, lst
    return lst

  ## Extract the best paths from i
  def getPaths(self, i):
    paths = {}
    if not i in self.pre:
      return paths
    for j in self.g.nodes():
      if not j in self.pre[i]:
        continue
      path = self.getPath(i, j)
      if path:
        paths[j] = path
    return paths

