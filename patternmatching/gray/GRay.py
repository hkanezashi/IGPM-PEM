"""
Extract a pattern subgraph with G-Ray algorithm

Tong, Hanghang, et al. "Fast best-effort pattern matching in large attributed graphs."
Proceedings of the 13th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2007.
"""


import networkx as nx
import random
from math import log
import logging

import rwr


class GRay:
  def __init__(self, graph, query):
    self.graph = graph
    self.query = query
    self.result = nx.Graph()
  
  
  def run_gray(self, num):
    """
    Run G-Ray algorithm for a pattern
    :param num: Number of iterations to find vertices
    :return:
    """
    
    for t in range(num):
      logging.info("---- Start itetation number " + str(t) + "/" + str(num) + " ----")
      touched = []
      nodemap = {}  ## Query Vertex -> Graph Vertex
      k = self.query.nodes()[0]
      i = self.seed_finder(k)
      if i is None:
        logging.warning("No more vertices available. Exit G-Ray algorithm.")
        return
      
      il = self.get_node_label(self.graph, i)
      nodemap[k] = i
      self.result.add_node(i, label=il)
      logging.info("## Mapping node: " + str(k) + " -> " + str(i))
      touched.append(k)
      unprocessed = nx.Graph(self.query)
      while unprocessed.size() > 0:  # unprocessed edges exist
        
        k = random.choice(touched)
        kl = self.get_node_label(self.query, k)
        
        for l in self.query.neighbors(k):
          if not unprocessed.has_edge(k, l):
            continue
          touched.append(l)
          ll = self.get_node_label(self.query, l)
          
          logging.info("Find the next vertex " + str(k) + "[" + kl + "] -> " + str(l) + "[" + ll + "]")
          
          if l in nodemap:
            j = nodemap[l]
          else:
            j = self.neighbor_expander(i, k, l, unprocessed) # find j(l) from i(k)
            if j is None:
              logging.warning("No more vertices available. Exit G-Ray algorithm.")
              for k, i in nodemap.iteritems():
                self.result.remove_node(i)  # Remove intermediate vertices
              return
            nodemap[l] = j
          
          logging.info("## Mapping node: " + str(l) + " -> " + str(j))
          if i == j:
            continue
          
          path = self.bridge(i, j)
          logging.debug("## Find path from " + str(i) + " -> " + str(j) + ": " + " ".join(str(n) for n in path))
          self.result.add_node(j)
          prev = i
          for n in path:
            self.result.add_edge(prev, n)
            prev = n
          unprocessed.remove_edge(k, l)
        
        
        # Find the next vertex
        k = l
        i = j
  
  
  def get_result(self):
    return self.result
  
  
  def seed_finder(self, k):
    logging.info("## Seed finder from: " + str(k))
    kl = self.get_node_label(self.query, k)  # Label of source
    rwrs = {}
    
    for l in self.query.neighbors(k):
      if l != k:
        rwrs[l] = self.rwr(self.query, l, k)
    logging.debug("#### SeedFinder#RWR: " + " ".join(rwrs))
    
    max_good = float('-inf')
    nodes = self.graph.nodes()
    seed = None
    for i in nodes:
      ## Exclude node which is already extracted
      if self.result.has_node(i) or self.get_node_label(self.graph, i) != kl:
        continue
      ## log goodness score
      log_good = 0
      neighbors = self.query.neighbors(k)
      for l in neighbors:
        # label = self.query.node[l]['label']
        # num = len([a for a,attr in self.graph.nodes_iter()]) if attr['label']==label])
        ll = self.get_node_label(self.query, l)  # Label of destination
        num = len(neighbors)
        if l != k:
          sum = 0
          for j in nodes:
            if self.get_node_label(self.graph, j) == ll:
              sum += log(self.rwr(self.graph, j, i) / num)
          log_good += sum / rwrs[l]
      
      logging.debug("#### SeedFinder#log_good: " + str(i) + " " + str(log_good))
      if log_good > max_good:
        max_good = log_good
        seed = i
        
    logging.info("SeedFinder#return: " + str(seed) + "!")
    return seed
  
  
  def neighbor_expander(self, i, k, l, unproc):
    kl = self.get_node_label(self.query, k)  # Label of source
    ll = self.get_node_label(self.query, l)  # Label of destination
    
    logging.info("## Neighbor expander from: " + str(i) + "[" + str(k) + "]")
    max_good = float('-inf')
    candidates_i = [i_ for i_ in self.graph.nodes() if self.get_node_label(self.graph, i_) == kl]
    candidates_j = [j_ for j_ in self.graph.nodes() if self.get_node_label(self.graph, j_) == ll]
    j = None
    
    for j_ in candidates_j:
      if j_ in self.result.nodes() or j_ == i:
        continue
      
      log_good = log(self.rwr(self.graph, i, j_))
      
      logging.debug("#### NeighborExpander#log_good: " + str(i) + " -> " + str(j_) + " " + str(log_good))
      
      if log_good > max_good:
        j = j_
        max_good = log_good
          
    logging.info("## NeighborExpander#return: " + str(j) + "!")
    return j
  
  
  def bridge(self, i, j):
    ## print i, j
    assert(i != j)
    logging.info("## Bridge between: " + str(i) + " -> " + str(j))
    V = self.graph.nodes()
    X = {i}
    d = {}
    l = {}
    pre = dict()
    d[i] = self.rwr(self.graph, i, i)
    l[i] = 1
    pre[i] = i
    for u in V:
      if i != u:
        d[u] = 0
        l[u] = 0
    
    while V:
      logging.debug("V list: " + " ".join(str(n) for n in V))
      maxd = 0
      u = V[0]
      for u_ in V:
        if d[u_] > maxd:
          maxd_ = d[u_]
          u = u_
      V.remove(u)
      X.add(u)
      logging.debug("Picked up: " + str(u))
      for v in self.graph.neighbors(u):
        logging.debug("Neighbor: " + str(v) + " V:" + " ".join(str(n) for n in V) + " result:" + " ".join(str(n) for n in self.result.nodes()))
        if (v in V):
          rw = self.rwr(self.graph, i, v)
          dist = (rw + d[u] * l[u])/(l[u] + 1)
          logging.debug("v=" + str(v) + " rwr=" + str(rw) + " d[u]=" + str(d[u]) + " l[u]=" + str(l[u]) + " dist=" + str(dist))
          if d[v] < dist:
            d[v] = dist
            l[v] = l[u] + 1
            pre[v] = u
            logging.debug("pre[" + str(v) + "] = " + str(u))
    lst = []
    v = j
    logging.debug("Pre: " + str(pre))
    logging.debug("d: " + str(d))
    logging.debug("l: " + str(l))
    while not v == i:
      lst.insert(0, v)
      v = pre[v]
    logging.info("## Bridge#return: " + " ".join(str(n) for n in lst) + "!")
    return lst  ## will not append the first element (i)
  
  def get_node_label(self, g, i):
    return g.node[i].get('label', 'white')
  
  def rwr(self, g, m, n):  # Random walk with restart m -> n in g
    RESTART_PROB = 0.7
    OG_PROB = 0.1
    rw = rwr.RWR(g)
    results = rw.run_exp(m, RESTART_PROB, OG_PROB)
    logging.debug("RWR: " + str(m) + " -> " + str(n) + " " + str(results))
    return results[n]



