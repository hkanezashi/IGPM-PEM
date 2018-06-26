"""
Extract multiple pattern subgraphs with G-Ray algorithm

Tong, Hanghang, et al. "Fast best-effort pattern matching in large attributed graphs."
Proceedings of the 13th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2007.
"""

from math import log
import time

from patternmatching.gray import rwr, extract
from patternmatching.query.Condition import *
from patternmatching.query import QueryResult


def equal_graphs(g1, g2):
  ns1 = set(g1.nodes())
  ns2 = set(g2.nodes())
  diff = ns1 ^ ns2
  if diff:  ## Not empty (has differences)
    return False
  
  """
  es1 = set(g1.edges())
  es2 = set(g2.edges())
  diff = es1 ^ es2
  if diff:
    return False
  """
  for n in ns1:
    ne1 = g1[n]
    ne2 = g2[n]
    # print n, ne1, ne2
    if ne1 != ne2:
      return False
  
  return True


def valid_result(result, query, nodemap):
  ## TODO: The number of vertices and edges of graphs with paths will vary
  hasPath = False
  etypes = nx.get_edge_attributes(query, TYPE)
  for k, v in etypes.iteritems():
    if v == PATH:
      hasPath = True
      break
  
  if not hasPath:
    nr_num = result.number_of_nodes()
    nq_num = query.number_of_nodes()
    if nr_num != nq_num:
      return False
    
    er_num = result.number_of_edges()
    eq_num = query.number_of_edges()
    if er_num != eq_num:
      return False

  # print nodemap
  # print query.edges()
  # print result.edges()
  for qn, rn in nodemap.iteritems():
    qd = query.degree(qn)
    rd = result.degree(rn)
    # print "degree:", qn, qd, rn, rd
    if qd != rd:
      return False
  
  return True


RESTART_PROB = 0.7
OG_PROB = 0.1

class GRayMultiple:
  """
  Class of basic G-Ray implementation (it outputs multiple patterns)
  """
  
  def __init__(self, graph, query, directed, cond, time_limit):
    self.graph = graph
    self.graph_rwr = rwr.RWR_WCC(graph, RESTART_PROB, OG_PROB)
    self.query = query
    self.directed = directed
    self.results = dict() ## Seed ID, QueryResult
    self.current_seed = None  # Current seed ID
    self.count = 0
    self.extracts = {}
    self.cond = cond ## Complex condition
    self.time_limit = time_limit
    self.called = 0

  def is_target(self):
    return self.current_seed == -1
  
  def process_gray(self):
    logging.debug("#### Find Seeds")
    k = list(self.query.nodes())[0]
    kl = Condition.get_node_label(self.query, k)
    kp = Condition.get_node_props(self.query, k)
    seeds = Condition.filter_nodes(self.graph, kl, kp)  # Find all candidates
    # seeds = [s for s in self.graph.nodes() if self.get_node_label(self.graph, s) == kl]  # Find all candidates
    if not seeds:  ## No seed candidates
      logging.debug("No more seed vertices available. Exit G-Ray algorithm.")
      return
    else:
      print("Number of seeds: %d" % len(seeds))

    st = time.time()  # Start time
    for i in seeds:
      self.called = 0
      self.current_seed = i
      
      if self.is_target():
        logging.info("#### Choose Seed: " + str(i) + " " + str(len(self.graph[i])))
      result = nx.MultiDiGraph() if self.directed else nx.MultiGraph()

      touched = []
      nodemap = {}  ## Query Vertex -> Graph Vertex
      unprocessed = self.query.copy()
      # unprocessed = nx.MultiDiGraph(self.query) if self.directed else nx.MultiGraph(self.query)

      il = Condition.get_node_label(self.graph, i)
      props = Condition.get_node_props(self.graph, i)
      nodemap[k] = i
      result.add_node(i)
      result.nodes[i][LABEL] = il
      for name, value in props.iteritems():
        result.nodes[i][name] = value

      # logging.debug("## Mapping node: " + str(k) + " : " + str(i))
      touched.append(k)
      
      # Start neighbor-expander and bridge
      self.process_neighbors(result, touched, nodemap, unprocessed)
      
      # print("Called %d times from %s with degree %d" % (self.called, str(i), self.graph.degree(i)))
      
      if 0.0 < self.time_limit < time.time() - st:
        print("Timeout G-Ray iterations")
        break


  def run_gray(self):
    logging.info("---- Start G-Ray ----")
    st = time.time()
    self.computeRWR()
    # if 14896 in self.graph:
    #   print 14896, self.graph_rwr.mat[14896]
    ed = time.time()
    logging.info("#### Compute RWR: %f [s]" % (ed - st))

    st = time.time()
    ext = extract.Extract(self.graph, self.graph_rwr)
    ext.computeExtract()
    self.extracts[''] = ext
    ed = time.time()
    logging.info("#### Compute Paths: %f [s]" % (ed - st))
    
    # pr = cProfile.Profile()
    # pr.enable()
    st = time.time()
    self.process_gray()
    ed = time.time()
    logging.info("#### Compute G-Ray: %f [s]" % (ed - st))
    # pr.disable()
    # stats = pstats.Stats(pr)
    # stats.sort_stats('tottime')
    # stats.print_stats()

  
  def getExtract(self, label):
    if not label in self.extracts:
      ext = extract.Extract(self.graph, self.graph_rwr, label)
      ext.computeExtract()
      self.extracts[label] = ext
    return self.extracts[label]

  def append_results(self, result, nodemap):
    if self.cond is not None and not self.cond.eval(result, nodemap):
      logging.debug("Invalid subgraph")
      return False  ## Not satisfied with complex condition
    
    ## Remove duplicates
    for r in self.results.values():
      rg = r.get_graph()
      if equal_graphs(rg, result):
        logging.debug("Duplicated subgraph " + str(result.nodes()))
        return False
    
    ## Register the result pattern
    qresult = QueryResult.QueryResult(result, nodemap)
    self.results[self.current_seed] = qresult  # Register QueryResult of current seed

    logging.debug("Result nodes:" + str(result.nodes()))
    return True

  
  def process_neighbors(self, result, touched, nodemap, unproc):
    self.called += 1
    if unproc.number_of_edges() == 0:
      if valid_result(result, self.query, nodemap):
        logging.debug("###### Found pattern " + str(self.count))
        if self.append_results(result, nodemap):
          self.count += 1
        return
      else:
        logging.debug("No more edges available. Exit G-Ray algorithm.")
        return
    if result.number_of_edges() > self.query.number_of_edges():
      logging.debug("Too many edges. Exit G-Ray algorithm.")
      return
    
    k = None
    l = None
    reversed_edge = False
    kl = None
    for k_ in touched:
      kl = Condition.get_node_label(self.query, k_)
      # kp = Condition.get_node_props(self.query, k_)
      
      ## Forward Edge
      for l_ in self.query.neighbors(k_):
        # print k_, kl, kp, l_, unproc.edges()
        if not unproc.has_edge(k_, l_):
          continue
        l = l_
        break
      if l is not None:
        k = k_
        # logging.debug("#### Process neighbors " + str(k) + " -> " + str(l))
        break
      
      ## Reversed Edge
      if self.directed:
        for l_ in self.query.predecessors(k_):
          if not unproc.has_edge(l_, k_):
            continue
          l = l_
          break
        if l is not None:
          reversed_edge = True
          k = k_
          # logging.debug("#### Process neighbors " + str(k) + " <- " + str(l))
          break
          
    
    if l is None:
      logging.info("No more vertices with the same label available. Exit G-Ray algorithm.")
      return
    
    if self.is_target():
      logging.info("#### Start Processing Neighbors from " + str(k) + " count " + str(self.count))
      logging.info("## result: " + " ".join([str(e) for e in result.edges()]))
      logging.info("## touchd: " + " ".join([str(n) for n in touched]))
      logging.info("## nodemp: " + str(nodemap))
      logging.info("## unproc: " + " ".join([str(e) for e in unproc.edges()]))
    
    i = nodemap[k]
    touched.append(l)
    ll = Condition.get_node_label(self.query, l)
    # lp = Condition.get_node_props(self.query, l)
    logging.debug("## Find the next vertex " + str(k) + "[" + kl + "] -> " + str(l) + "[" + ll + "]")
    
    is_path = False
    if not reversed_edge and Condition.is_path(self.query, k, l) or reversed_edge and Condition.is_path(self.query, l, k):
      is_path = True
    
    #### Find a path or edge (Begin)
    src, dst = (l, k) if reversed_edge else (k, l)
    
    elabel = Condition.get_edge_label(unproc, src, dst)
    # print elabel
    if elabel is None:  # Any label is OK
      eid = None
      el = ''
    else:
      eid = elabel[0]
      el = elabel[1]
    Condition.remove_edge_from_id(unproc, src, dst, eid)
    
    
    if is_path:
      paths = self.getExtract(el).getPaths(i)
      if not paths:
        logging.info("No more paths available. Exit G-Ray algorithm.")
        return
      for j, path in paths.iteritems():
        result_ = nx.MultiDiGraph(result) if self.directed else nx.MultiGraph(result)
        touched_ = list(touched)
        nodemap_ = dict(nodemap)
        unproc_ = nx.MultiDiGraph(unproc) if self.directed else nx.MultiGraph(unproc)
    
        if l in nodemap_ and nodemap_[l] != j:
          prevj = nodemap_[l]
          result_.remove_node(prevj)
    
        nodemap_[l] = j
        props = Condition.get_node_props(self.graph, j)
        result_.add_node(j)
        for key, value in props.iteritems():
          result_.nodes[j][key] = value
        
        prev = i
        for n in path:
          result_.add_edge(prev, n)
          prev = n
        self.process_neighbors(result_, touched_, nodemap_, unproc_)
    
    else:
      if l in nodemap:
        jlist = [nodemap[l]]
      else:
        jlist = self.neighbor_expander(i, k, l, result, reversed_edge)  # find j(l) from i(k)
        if not jlist:  ## No more neighbor candidates
          logging.debug("No more neighbor vertices available. Exit G-Ray algorithm.")
          return
      
      for j in jlist:
        g_src, g_dst = (j, i) if reversed_edge else (i, j)
        if g_src == g_dst:  ## No bridge process necessary
          continue
        path = self.bridge(g_src, g_dst)
        # if len(path) != 1:
        if not path:
          logging.debug("## No path found: " + str(g_src) + " -> " + str(g_dst))
          continue
        logging.debug("## Find path from " + str(g_src) + " -> " + str(g_dst) + ": " + " ".join(str(n) for n in path))
        
        result_ = nx.MultiDiGraph(result) if self.directed else nx.MultiGraph(result)
        touched_ = list(touched)
        nodemap_ = dict(nodemap)
        unproc_ = nx.MultiDiGraph(unproc) if self.directed else nx.MultiGraph(unproc)
    
        if l in nodemap_ and nodemap_[l] != j:  ## Need to replace mapping
          prevj = nodemap_[l]
          result_.remove_node(prevj)
        nodemap_[l] = j
        props = Condition.get_node_props(self.graph, j)
        result_.add_node(j)
        # print props
        for k, v in props.iteritems():
          result_.nodes[j][k] = v
        # print result_.nodes(data=True)[j]
        
        prev = g_src
        valid = True
        for n in path:
          # print path, prev, n
          if not Condition.has_edge_label(self.graph, prev, n, el):
            valid = False
            break
          result_.add_edge(prev, n)
          prev = n
        if valid:
          self.process_neighbors(result_, touched_, nodemap_, unproc_)
    #### Find a path or edge (End)
    
  ## List of tuple (extracted graph and node map)
  def get_results(self):
    return self.results
  
  def seed_finder(self, k):
    logging.debug("## Seed finder from: " + str(k))
    kl = Condition.get_node_label(self.query, k)  # Label of source
    kp = Condition.get_node_props(self.query, k)  # Props of source
    rwrs = {}
    
    for l in self.query.neighbors(k):
      if l != k:
        rwrs[l] = self.rwr(self.query, l, k)
    # logging.debug("#### SeedFinder#RWR: " + str(rwrs))
    
    max_good = float('-inf')
    nodes = self.graph.nodes()
    seeds = []
    
    for i in nodes:
      ## Exclude node which is already extracted
      #if Condition.get_node_label(self.graph, i) != kl:
      if not Condition.satisfies_node(self.graph, i, kl, kp):
        continue
      
      log_good = 0
      neighbors = self.query.neighbors(k)
      for l in neighbors:
        ll = Condition.get_node_label(self.query, l)  # Label of destination
        lp = Condition.get_node_props(self.query, l)  # Props of destination
        num = len(neighbors)
        if l != k:
          log_sum = 0
          for j in nodes:
            if Condition.satisfies_node(self.graph, j, ll, lp):
              log_sum += log(self.getRWR(j, i) / num)
          log_good += log_sum / rwrs[l]
      
      # logging.debug("#### SeedFinder#log_good: " + str(i) + " " + str(log_good) + " max_good: " + str(max_good))
      if log_good > max_good:
        # logging.debug("#### Found max log_good")
        max_good = log_good
        seeds = [i]  ## Reset seed candidates
      elif log_good >= max_good - 1.0e-5:  ## Almost same, may be a little smaller due to limited precision
        # logging.debug("#### Found max log_good")
        seeds.append(i)  ## Add seed candidates
      
    logging.debug("SeedFinder#return: " + " ".join([str(seed) for seed in seeds]) + "!")
    return seeds
  
  def neighbor_expander(self, i, k, l, result, reversed_edge):
    # kl = Condition.get_node_label(self.query, k)  # Label of source
    ll = Condition.get_node_label(self.query, l)  # Label of destination
    
    logging.debug("## Neighbor expander from: " + str(i) + "[" + str(k) + "]")
    max_good = float('-inf')
    candidates_j = Condition.filter_nodes(self.graph, ll, {})
    # candidates_i = [i_ for i_ in self.graph.nodes() if self.get_node_label(self.graph, i_) == kl]
    # candidates_j = [j_ for j_ in self.graph.nodes() if self.get_node_label(self.graph, j_) == ll]
    j = []
    # print "candidates", candidates_j
    # print "result nodes", result.nodes()
    
    candidates_j = self.get_connected(i) & set(candidates_j)
    
    for j_ in candidates_j:
      if j_ in result.nodes() or j_ == i:
        continue
      
      if reversed_edge:
        log_good = log(self.getRWR(j_, i) + 1.0e-10)  # avoid math domain errors when the vertex is unreachable
        # logging.debug("#### NeighborExpander#log_good: " + str(i) + " <- " + str(j_) + " " + str(log_good))
      else:
        log_good = log(self.getRWR(i, j_) + 1.0e-10)  # avoid math domain errors when the vertex is unreachable
        # logging.debug("#### NeighborExpander#log_good: " + str(i) + " -> " + str(j_) + " " + str(log_good))
      
      if log_good > max_good:
        j = [j_]
        max_good = log_good
      elif log_good >= max_good - 1.0e-5:  ## Almost same, may be a little smaller due to limited precision
        j.append(j_)
      
    logging.debug("## NeighborExpander#return: " + " ".join([str(j_) for j_ in j]) + "!")
    return j
  
  
  def bridge(self, i, j, label=None):
    if label is None:
      label = ''
    return self.getExtract(label).getPath(i, j)

  def computeRWR_batch(self):
    # rw = rwr.RWR_WCC(self.graph, RESTART_PROB, OG_PROB)
    # self.graph_rwr = rw.rwr_all()
    # rw = rwr.RWR(self.graph)
    # for m in self.graph.nodes():
    #   results = rw.run_exp(m, RESTART_PROB, OG_PROB)
    #   self.graph_rwr[m] = results
    self.computeRWR()
    # self.graph_rwr.rwr_all()
  
  
  def computeRWR(self):
    self.graph_rwr.rwr_set(self.graph.nodes())
    # st = time.time()  # Start time
    # for m in self.graph.nodes():
    #   self.graph_rwr.rwr_single(m)
    #   if 0.0 < self.time_limit < time.time() - st:
    #     print("Timeout RWR iterations")
    #     break
  
  def getRWR(self, m, n):
    # if not m in self.graph_rwr:
    #   return 0.0
    # else:
    #   return self.graph_rwr[m].get(n, 0.0)
    return self.graph_rwr.get_value(m, n)
  
  def get_connected(self, n):
    return set(self.graph_rwr.get_dsts(n))
  
  #def getExtract(self):
  #  return self.extract
  
  @staticmethod
  def rwr(g, m, n):  # Random walk with restart m -> n in g
    rw = rwr.RWR(g)
    results = rw.run_exp(m, RESTART_PROB, OG_PROB)
    logging.debug("RWR: " + str(m) + " -> " + str(n) + " " + str(results))
    return results[n]



