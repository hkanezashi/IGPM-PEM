"""
Random walk with restart with matrix multiplications

Retrieved and simplified some functions from https://github.com/TuftsBCB/Walker/blob/master/walker.py
"""


import numpy as np
import networkx as nx
from sklearn.preprocessing import normalize
from collections import defaultdict
import sys
import pickle

CONV_THRESHOLD = 0.001


class RWR_WCC:
  """RWR optimized with weakly connected component
  """
  
  def __init__(self, g, restart_prob, og_prob):
    self.g = g
    self.restart_prob = restart_prob
    self.og_prob = og_prob
    self.wccs = list(nx.weakly_connected_components(g.to_directed()))
    self.num = g.number_of_nodes()
    # self.idmap = dict()
    # for idx, vid in enumerate(g.nodes()):
    #   self.idmap[vid] = idx  # Vertex ID --> Index
    # self.mat = np.zeros((num, num), dtype=float)
    self.mat = dict()
  
  @staticmethod
  def load_pickle(fname):
    with open(fname, mode="rb") as rf:
      data = pickle.load(rf)
    g = data["graph"]
    restart_prob = data["restart_prob"]
    og_prob = data["og_prob"]
    mat = data["mat"]
    
    rwr_wcc = RWR_WCC(g, restart_prob, og_prob)
    rwr_wcc.mat = mat
    return rwr_wcc

  def dump_pickle(self, fname):
    data = dict()
    data["graph"] = self.g
    data["restart_prob"] = self.restart_prob
    data["og_prob"] = self.og_prob
    data["mat"] = self.mat
    with open(fname, mode="wb") as wf:
      pickle.dump(data, wf)
  
  def add_edges(self, edges):
    self.g.add_edges_from(edges)
    self.wccs = list(nx.weakly_connected_components(self.g.to_directed()))
    self.num = self.g.number_of_nodes()
    
  
  def rwr_single(self, src):
    for wcc in self.wccs:
      if src in wcc:
        g_ = nx.subgraph(self.g, wcc)
        r_ = RWR(g_)
        ret = r_.run_exp(src, self.restart_prob, self.og_prob)
        self.set_values(src, ret)
        break
  
  def rwr_set(self, nodes):
    remain = set(nodes)
    # count = 0
    for wcc in self.wccs:
      found = set(wcc)
      rc_set = remain & found
      if rc_set:
        g_ = nx.subgraph(self.g, wcc)
        r_ = RWR(g_)
        for src in rc_set:
          # count += 1
          ret = r_.run_exp(src, self.restart_prob, self.og_prob)
          self.set_values(src, ret)
          # if count % 100 == 0:
          #   print count
        remain -= rc_set
        if not remain:
          return
          
  
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
    self.restart_prob = 0.7
    self.og_prob = 0.1
  
  def _build_matrices(self, graph):
    self.OG = graph
    self.nodelist = list(self.OG.nodes())
    og_not_normalized = nx.to_numpy_matrix(graph)
    self.og_matrix = self._normalize_cols(og_not_normalized)
  
  @staticmethod
  def _normalize_cols(og_not_normalized):
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
    p_0 = [0.0] * self.OG.number_of_nodes()
    for source_id in sources:
      try:
        source_index = self.nodelist.index(source_id)
        p_0[source_index] = 1.0 / len(sources)
      except ValueError:
        sys.exit("Source node {} is not in original graph. Source: {}. Exiting.".format(source_id, sources))
    return np.array(p_0)


def run_karate():
  # g = nx.karate_club_graph()
  g = nx.Graph()
  g.add_edges_from([(1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 5)])
  
  r = RWR(g)
  nodes = g.nodes()
  for n in nodes:
    results = r.run_exp(n, 0.7, 0.1)
    print(n, results)
  
  r = RWR_WCC(g, 0.7, 0.1)
  r.rwr_all()
  for i in g.nodes():
    for j in g.nodes():
      print(i, j, r.get_value(i, j))
  

if __name__ == "__main__":
  run_karate()
  
  
  
