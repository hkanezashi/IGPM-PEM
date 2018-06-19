"""
Random walk with restart with matrix multiplications

Retrieved and simplified some functions from https://github.com/TuftsBCB/Walker/blob/master/walker.py
"""


import numpy as np
import networkx as nx
from sklearn.preprocessing import normalize
from collections import defaultdict
import sys

CONV_THRESHOLD = 0.000001


class RWR_WCC:
  """RWR optimized with weakly connected component
  """
  
  def __init__(self, g, restart_prob, og_prob):
    self.g = g
    self.restart_prob = restart_prob
    self.og_prob = og_prob
    self.wccs = nx.weakly_connected_components(g.to_directed())
    self.num = g.number_of_nodes()
    self.idmap = dict()
    for idx, vid in enumerate(g.nodes()):
      self.idmap[vid] = idx  # Vertex ID --> Index
    # self.mat = np.zeros((num, num), dtype=float)
    self.mat = dict()
  
  def rwr_single(self, src):
    # src_i = self.idmap[src]
    for wcc in self.wccs:
      if src in wcc:
        g_ = nx.subgraph(self.g, wcc)
        r_ = RWR(g_)
        ret = r_.run_exp(src, self.restart_prob, self.og_prob)
        # for dst, value in ret.iteritems():
        #   dst_i = self.idmap[dst]
        #   self.set_value(src_i, dst_i, value)
        self.set_values(src, ret)
  
  def rwr_all(self):
    for wcc in self.wccs:
      g_ = nx.subgraph(self.g, wcc)
      r_ = RWR(g_)
      for src in wcc:
        ret = r_.run_exp(src, self.restart_prob, self.og_prob)
        self.set_values(src, ret)
        # src_i = self.idmap[src]
        # for dst, value in ret.iteritems():
        #   dst_i = self.idmap[dst]
        #   # self.mat[src_i, dst_i] = value
        #   self.set_value(src_i, dst_i, value)
  
  # def rwr_single(self, n):
  #   for wcc in self.wccs:
  #     if n in wcc:
  #       g_ = nx.subgraph(self.g, wcc)
  #       r_ = RWR(g_)
  #       return r_.run_exp(n, self.restart_prob, self.og_prob)
  #   return dict()
  #
  # def rwr_all(self):
  #   result = dict()
  #   for wcc in self.wccs:
  #     g_ = nx.subgraph(self.g, wcc)
  #     r_ = RWR(g_)
  #     for n_ in wcc:
  #       ret = r_.run_exp(n_, self.restart_prob, self.og_prob)
  #       result[n_] = ret
  #   return result
  
  def get_dsts(self, src):
    if not src in self.mat:
      return set()
    else:
      return self.mat[src].keys()
  
  def set_values(self, src, value_map):
    if not src in self.mat:
      d = defaultdict(float)
      # d.update(dict.fromkeys(self.g.nodes(), 0.0))
      self.mat[src] = d
    self.mat[src].update(value_map)
  
  def set_value(self, src, dst, value):
    if not src in self.mat:
      d = defaultdict(float)
      # d.update(dict.fromkeys(self.g.nodes(), 0.0))
      self.mat[src] = d
    self.mat[src][dst] = value
  
  def get_value(self, src, dst):
    # src_i = self.idmap[src]
    # dst_i = self.idmap[dst]
    if not src in self.mat:
      d = defaultdict(float)
      # d.update(dict.fromkeys(self.g.nodes(), 0.0))
      self.mat[src] = d
      return 0.0
    else:
      return self.mat[src][dst]


class RWR:
  def __init__(self, graph):
    self._build_matrices(graph.to_directed().reverse()) ## TODO: edges need to be reversed.
  
  def _build_matrices(self, graph):
    self.OG = graph
    self.nodelist = list(self.OG.nodes())
    og_not_normalized = nx.to_numpy_matrix(graph)
    self.og_matrix = self._normalize_cols(og_not_normalized)
  
  def _normalize_cols(self, og_not_normalized):
    return normalize(og_not_normalized, norm='l1', axis=0)
  
  def run_exp(self, source, restart_prob, og_prob):
    self.restart_prob = restart_prob
    self.og_prob = og_prob
    
    p_0 = self._set_up_p0([source])
    diff_norm = 1
    p_t = np.copy(p_0)
    
    while diff_norm > CONV_THRESHOLD:
      p_t_1 = self._calculate_next_p(p_t, p_0)
      diff_norm = np.linalg.norm(np.subtract(p_t_1, p_t), 1)
      p_t = p_t_1
    
    result = {}  ## target, score (probability)
    for node, prob in self._generate_rank_list(p_t):
      result[node] = prob
    return result
  
  def _generate_rank_list(self, p_t):
    gene_probs = zip(self.OG.nodes(), p_t.tolist())
    for s in sorted(gene_probs, key=lambda x: x[1], reverse=True):
      yield s[0], s[1]
  
  def _calculate_next_p(self, p_t, p_0):
    epsilon = np.squeeze(np.asarray(np.dot(self.og_matrix, p_t)))
    no_restart = epsilon * (1 - self.restart_prob)
    restart = p_0 * self.restart_prob
    return np.add(no_restart, restart)
  
  def _set_up_p0(self, sources):
    p_0 = [0] * self.OG.number_of_nodes()
    for source_id in sources:
      try:
        source_index = self.nodelist.index(source_id)
        p_0[source_index] = 1 / float(len(sources))
      except ValueError:
        sys.exit("Source node {} is not in original graph. Source: {}. Exiting.".format(source_id, sources))
    # print p_0
    return np.array(p_0)


def run_karate():
  # g = nx.karate_club_graph()
  g = nx.Graph()
  g.add_edges_from([(1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 5)])
  
  r = RWR(g)
  nodes = g.nodes()
  for n in nodes:
    results = r.run_exp(n, 0.7, 0.1)
    print n, results
  
  r = RWR_WCC(g, 0.7, 0.1)
  r.rwr_all()
  for i in g.nodes():
    for j in g.nodes():
      print i, j, r.get_value(i, j)
  

if __name__ == "__main__":
  run_karate()
  
  
  
