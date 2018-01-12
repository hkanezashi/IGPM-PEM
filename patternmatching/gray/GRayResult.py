from math import log
from random import random

import networkx as nx
import RWR


class GRayResult:
  
  def __init__(self, touched, nodemap, unproc, input, query, result):
    self.touched = list(touched)
    self.nodemap = dict(nodemap)
    self.unproc = nx.Graph(unproc)
    self.input = input
    self.query = query
    self.result = nx.Graph(result)
  
  def edge_processed(self, i, j):
    return not self.unproc.has_edge(i, j)
  
  def finished(self):
    return self.unproc.number_of_edges() == 0
    
  def process_neighbors(self, i):
    if self.finished():
      return self.result
    
    k = random.choice(self.touched)
    kl = self.get_node_label(self.query, k)
    for l in self.query.neighbors(k):
      if self.edge_processed(k, l):
        continue
      self.touched.append(l)
      ll = self.get_node_label(self.query, l)
      
      #### Find Neighbors
      jlist = []
      if l in self.nodemap:
        jlist.append(self.nodemap[l])
      else:
        jlist = self.neighbor_expander(i, k, l)
        if not jlist:
          return None
      
      for j in jlist:
        if i == j:
          continue
        
        #### Bridge
        path = self.bridge(i, j)
        
        #### Find next neighbors for all candidates
        self.nodemap[l] = j
        self.result.add_node(j)
        prev = i
        for n in path:
          self.result.add_edge(prev, l)
          prev = n
        self.unproc.remove_edge(k, l)
        
        
        

  def neighbor_expander(self, i, k, l):
    ll = self.get_node_label(self.query, l)  # Label of destination
  
    max_good = float('-inf')
    candidates_j = [j_ for j_ in self.input.nodes() if self.get_node_label(self.input, j_) == ll]
    j = []
  
    for j_ in candidates_j:
      if j_ == i:
        continue
    
      log_good = log(self.rwr(self.input, i, j_))
      if log_good > max_good:
        j = [j_]
        max_good = log_good
      elif log_good == max_good:
        j.append(j_)
    return j

  def bridge(self, i, j):
    assert (i != j)
    V = self.graph.nodes()
    X = {i}
    d = {}
    l = {}
    pre = {}
    d[i] = self.rwr(self.graph, i, i)
    l[i] = 1
    pre[i] = i
    for u in V:
      if i != u:
        d[u] = 0
        l[u] = 0
  
    while V:
      maxd = 0
      u = V[0]
      for u_ in V:
        if d[u_] > maxd:
          maxd_ = d[u_]
          u = u_
      V.remove(u)
      X.add(u)
      for v in self.graph.neighbors(u):
        if (v in V):
          rw = self.rwr(self.graph, i, v)
          dist = (rw + d[u] * l[u]) / (l[u] + 1)
          if d[v] < dist:
            d[v] = dist
            l[v] = l[u] + 1
            pre[v] = u
    lst = []
    v = j
    while not v == i:
      lst.insert(0, v)
      v = pre[v]
    return lst  ## will not append the first element (i)

  def get_result(self):
    return self.result
  
  def get_node_label(self, g, i):
    return g.node[i].get('label', 'white')
  
  def rwr(self, g, m, n):  # Random walk with restart m -> n in g
    RESTART_PROB = 0.7
    OG_PROB = 0.1
    rw = RWR.RWR(g)
    results = rw.run_exp(m, RESTART_PROB, OG_PROB)
    return results[n]
