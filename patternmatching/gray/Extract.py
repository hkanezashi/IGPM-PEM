from Condition import *

MAX_LENGTH = 5

class Extract:
  
  def __init__(self, g, rwr, label=None):
    # self.d = {}
    # self.l = {}
    self.pre = {}
    self.rwr = rwr
    self.g = g
    self.label = label
  
  def getRWR(self, i, j):
    return self.rwr[i][j]
  
  def computeExtract(self):
    # self.computeRWR()
    for i in self.g.nodes():
      self.pre[i] = {}
      self.pre[i][i] = i
      self.computeExtractSingle(i)

  def computeExtractSingle(self, i):
    d = {}   ## Distance
    l = {}   ## Hops
    X = set()   ## Finished set
    # V = self.g.nodes() ## Processing queue
    # V = nx.ego_graph(self.g, i, radius=MAX_LENGTH).nodes()
    V = {i}
    d[i] = self.getRWR(i, i)
    l[i] = 1
    for u in V:
      if i != u:
        d[u] = 0
        l[u] = 0
    
    while V:
      max_d = 0
      u = None # V[0]
      for u_ in V:
        if d[u_] > max_d:
          max_d = d[u_]
          u = u_
      V.remove(u)
      X.add(u)
      
      if u in l:
        if l[u] >= MAX_LENGTH:
          continue
      else:
        l[u] = 0
      
      for v in self.g.neighbors(u):
        if self.label is not None and not self.label in Condition.get_edge_labels(self.g, u, v).values():
          continue
        if not v in X:
          V.add(v)
        rw = self.getRWR(i, v)
        lu = l[u]
        dist = (rw + d[u] * lu)/(lu + 1)
        if (not v in d) or (d[v] < dist):
          d[v] = dist
          l[v] = lu + 1
          self.pre[i][v] = u
  
  ## Extract the best path i -> j
  def getPath(self, i, j):
    lst = []
    if not i in self.pre:
      return []
    if not j in self.pre[i]:
      return []
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

