"""
Random walk with restart with matrix multiplications

Retrieved and simplified some functions from https://github.com/TuftsBCB/Walker/blob/master/walker.py
"""


import numpy as np
import networkx as nx
from sklearn.preprocessing import normalize
import logging
import sys

CONV_THRESHOLD = 0.000001


class RWR:
  def __init__(self, graph):
    self._build_matrices(graph.to_directed().reverse()) ## TODO: edges need to be reversed.
  
  def _build_matrices(self, graph):
    self.OG = graph
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
        source_index = self.OG.nodes().index(source_id)
        p_0[source_index] = 1 / float(len(sources))
      except ValueError:
        sys.exit("Source node {} is not in original graph. Source: {}. Exiting.".format(source_id, sources))
    return np.array(p_0)
