import networkx as nx
import random
from math import log
import logging

import RWR
import GRayResult
import Extract
from Condition import *
import QueryResult

class GRayMultiple:
  def __init__(self, graph, query, directed, cond):
    self.graph = graph
    self.graph_rwr = {}
    self.query = query
    self.directed = directed
    self.results = [] ## QueryResult
    self.count = 0
    self.extracts = {}
    self.cond = cond ## Complex condition
  
  def run_gray(self):
    logging.info("---- Start G-Ray ----")
    logging.info("#### Compute RWR")
    self.computeRWR()
    logging.info("#### Compute Extract")
    extract = Extract.Extract(self.graph, self.graph_rwr)
    extract.computeExtract()
    self.extracts[''] = extract

    logging.debug("#### Find Seeds")
    k = self.query.nodes()[0]
    kl = Condition.get_node_label(self.query, k)
    kp = Condition.get_node_props(self.query, k)
    seeds = Condition.filter_nodes(self.graph, kl, kp)  # Find all candidates
    # seeds = [s for s in self.graph.nodes() if self.get_node_label(self.graph, s) == kl]  # Find all candidates
    if not seeds:  ## No seed candidates
      logging.debug("No more seed vertices available. Exit G-Ray algorithm.")
      return
    
    for i in seeds:
      logging.debug("#### Choose Seed: " + str(i))
      result = nx.MultiDiGraph() if self.directed else nx.MultiGraph()
      # self.results.append(result)
      
      touched = []
      nodemap = {}  ## Query Vertex -> Graph Vertex
      unprocessed = self.query.copy()
      # unprocessed = nx.MultiDiGraph(self.query) if self.directed else nx.MultiGraph(self.query)
      
      il = Condition.get_node_label(self.graph, i)
      props = Condition.get_node_props(self.graph, i)
      nodemap[k] = i
      # result.add_node(i, label=il)
      result.add_node(i, props)
      # logging.debug("## Mapping node: " + str(k) + " : " + str(i))
      touched.append(k)
      
      self.process_neighbors(result, touched, nodemap, unprocessed)
  
  def getExtract(self, label):
    if not label in self.extracts:
      extract = Extract.Extract(self.graph, self.graph_rwr, label)
      extract.computeExtract()
      self.extracts[label] = extract
    return self.extracts[label]
  
  def equal_graphs(self, g1, g2):
    ns1 = set(g1.nodes())
    ns2 = set(g2.nodes())
    # logging.debug("Node 1: " + str(ns1))
    # logging.debug("Node 2: " + str(ns2))
    diff = ns1 ^ ns2
    if diff:  ## Not empty (has differences)
      return False
    
    es1 = set(g1.edges())
    es2 = set(g2.edges())
    # logging.debug("Edge 1: " + str(es1))
    # logging.debug("Edge 2: " + str(es2))
    diff = es1 ^ es2
    if diff:
      return False
    
    return True

  def valid_result(self, result, query, nodemap):
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
    
    """
    r_deg = sorted(result.degree(result.nodes()).values())
    q_deg = sorted(query.degree(query.nodes()).values())
    # print r_deg, q_deg
    if r_deg != q_deg:
      return False
    """
    return True

  def append_results(self, result, nodemap):
    if self.cond is not None and not self.cond.eval(result, nodemap):
      return False  ## Not satisfied with complex condition
    
    for r in self.results:
      rg = r.get_graph()
      if self.equal_graphs(rg, result):
        return False
    qresult = QueryResult.QueryResult(result, nodemap)
    self.results.append(qresult)
    return True

  """
  Remove a edge (i -> j) with specified label
  The argument 'key' is used as label
  https://networkx.github.io/documentation/networkx-1.9/reference/generated/networkx.MultiGraph.remove_edge.html
  """
  """
  @staticmethod
  def remove_edge_from_label(g, i, j, label):
    g.remove_edge(i, j, key=label)
    assert label != ''
    print g.edge[i][j], label
    for k, v in g.edge[i][j].iteritems():
      if v[LABEL] == label:
        del g.edge[i][j][k]
        # print g.edge[i][j]
        return
  """
  
  def process_neighbors(self, result, touched, nodemap, unproc):
    if unproc.number_of_edges() == 0:
      if self.valid_result(result, self.query, nodemap):
        logging.debug("###### Found pattern " + str(self.count))
        if self.append_results(result, nodemap):
          self.count += 1
        return
      else:
        logging.debug("No more edges available. Exit G-Ray algorithm.")
        return
    
    k = None
    l = None
    reversed_edge = False
    for k_ in touched:
      kl = Condition.get_node_label(self.query, k_)
      kp = Condition.get_node_props(self.query, k_)
      
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
      logging.debug("No more vertices with the same label available. Exit G-Ray algorithm.")
      return
    
    logging.debug("#### Start Processing Neighbors from " + str(k) + " count " + str(self.count))
    logging.debug("## result: " + " ".join([str(e) for e in result.edges()]))
    logging.debug("## touchd: " + " ".join([str(n) for n in touched]))
    logging.debug("## nodemp: " + str(nodemap))
    logging.debug("## unproc: " + " ".join([str(e) for e in unproc.edges()]))
    
    i = nodemap[k]
    touched.append(l)
    ll = Condition.get_node_label(self.query, l)
    lp = Condition.get_node_props(self.query, l)
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
        logging.debug("No more paths available. Exit G-Ray algorithm.")
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
        result_.add_node(j, props)
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
        result_.add_node(j, props)
        
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
    
    
    
    """
    #### Find Paths
    if reversed_edge and Condition.is_path(self.query, l, k):
      elabel = Condition.get_edge_label(unproc, l, k)
      print elabel
      if not elabel:
        logging.debug("No more paths available. Exit G-Ray algorithm.")
        return
      eid = elabel[0]
      el = elabel[1]
      Condition.remove_edge_from_id(unproc, l, k, eid)
      #unproc.remove_edge(l, k)
      paths = self.getExtract(el).getPaths(i)
      if not paths:
        logging.debug("No more paths available. Exit G-Ray algorithm.")
        return
      for j, path in paths.iteritems():
        result_ = nx.MultiDiGraph(result)
        touched_ = list(touched)
        nodemap_ = dict(nodemap)
        unproc_ = nx.MultiDiGraph(unproc)
        
        if l in nodemap_ and nodemap_[l] != j:
          prevj = nodemap_[l]
          result_.remove_node(prevj)
        
        nodemap_[l] = j
        props = Condition.get_node_props(self.graph, j)
        result_.add_node(j, props)
        prev = j
        for n in path:
          result_.add_edge(prev, n)
          prev = n
        
        #### Find next neighbors for all candidates
        self.process_neighbors(result_, touched_, nodemap_, unproc_)
      return
    
    elif Condition.is_path(self.query, k, l):
      elabel = Condition.get_edge_label(unproc, k, l)
      print elabel
      if not elabel:
        logging.debug("No more paths available. Exit G-Ray algorithm.")
        return
      eid = elabel[0]
      el = elabel[1]
      Condition.remove_edge_from_id(unproc, k, l, eid)
      # el = Condition.get_edge_label(unproc, k, l)[1]
      # Condition.remove_edge_from_label(unproc, l, k, el)
      # unproc.remove_edge(k, l)
      paths = self.getExtract(el).getPaths(i)
      if not paths:
        logging.debug("No more paths available. Exit G-Ray algorithm.")
        return
      
      for j, path in paths.iteritems():
        result_ = nx.MultiDiGraph(result) if self.directed else nx.MultiGraph(result)
        touched_ = list(touched)
        nodemap_ = dict(nodemap)
        # unproc_ = nx.MultiDiGraph(unproc) if self.directed else nx.MultiGraph(unproc)
        unproc_ = unproc.copy()
        
        if l in nodemap_ and nodemap_[l] != j:
          prevj = nodemap_[l]
          result_.remove_node(prevj)
        
        nodemap_[l] = j
        props = Condition.get_node_props(self.graph, j)
        result_.add_node(j, props)
        prev = i
        for n in path:
          result_.add_edge(prev, n)
          prev = n
        
        #### Find next neighbors for all candidates
        self.process_neighbors(result_, touched_, nodemap_, unproc_)
      return
    
    
    
    #### Find Direct Neighbors
    if l in nodemap:
      jlist = [nodemap[l]]
    else:
      jlist = self.neighbor_expander(i, k, l, result, reversed_edge) # find j(l) from i(k)
      if not jlist:  ## No more neighbor candidates
        logging.debug("No more neighbor vertices available. Exit G-Ray algorithm.")
        return
    # print "jlist", jlist
    
    if reversed_edge:
      elabel = Condition.get_edge_label(unproc, l, k)
      print elabel
      eid = elabel[0]
      el = elabel[1]
      Condition.remove_edge_from_id(unproc, l, k, eid)
      # el = Condition.get_edge_label(unproc, l, k)[1]  ## Edge Label
      # Condition.remove_edge_from_label(unproc, l, k, el)
      # unproc.remove_edge(l, k)
      # logging.debug("## unproc: " + " ".join([str(e) for e in unproc.edges()]))
      for j in jlist:
        if i == j:  ## No bridge process necessary
          continue
        #if Condition.get_edge_label(self.graph, j, i) != el:
        #  continue
        # logging.debug("## Mapping node: " + str(l) + " : " + str(j))
  
        #### Bridge
        # print i, j
        path = self.bridge(j, i)
        # path = self.bridge(j, i, el)
        if not path:
          # logging.debug("## No paths found: " + str(i) + " <- " + str(j))
          continue
        # logging.debug("## Find path from " + str(i) + " <- " + str(j) + ": " + " ".join(str(n) for n in path))
        
        touched_ = list(touched)
        nodemap_ = dict(nodemap)
        unproc_ = nx.MultiDiGraph(unproc)
        result_ = nx.MultiDiGraph(result)
        
        if l in nodemap_ and nodemap_[l] != j:  ## Need to replace mapping
          prevj = nodemap_[l]
          result_.remove_node(prevj)
        
        nodemap_[l] = j
        props = Condition.get_node_props(self.graph, j)
        result_.add_node(j, props)
        prev = j
        for n in path:
          if not Condition.has_edge_label(self.graph, prev, j, el):
            continue
          result_.add_edge(prev, n)
          prev = n
        
        #### Find next neighbors for all candidates
        self.process_neighbors(result_, touched_, nodemap_, unproc_)
      
    else:
      elabel = Condition.get_edge_label(unproc, k, l)
      print elabel
      eid = elabel[0]
      el = elabel[1]
      Condition.remove_edge_from_id(unproc, k, l, eid)
      # el = Condition.get_edge_label(unproc, k, l)[1]  ## Edge Label
      # Condition.remove_edge_from_label(unproc, l, k, el)
      
      # logging.debug("## unproc: " + " ".join([str(e) for e in unproc.edges()]))
      for j in jlist:
        if i == j:  ## No bridge process necessary
          continue
        # logging.debug("## Mapping node: " + str(l) + " : " + str(j))
        
        #### Bridge
        # print i, j, el
        path = self.bridge(i, j)
        # path = self.bridge(i, j, el)
        if not path:
          logging.debug("## No paths found: " + str(i) + " -> " + str(j))
          continue
        logging.debug("## Find path from " + str(i) + " -> " + str(j) + ": " + " ".join(str(n) for n in path))
        
        result_ = nx.MultiDiGraph(result) if self.directed else nx.MultiGraph(result)
        touched_ = list(touched)
        nodemap_ = dict(nodemap)
        unproc_ = nx.MultiDiGraph(unproc) if self.directed else nx.MultiGraph(unproc)
        
        if l in nodemap_ and nodemap_[l] != j:  ## Need to replace mapping
          prevj = nodemap_[l]
          result_.remove_node(prevj)
        
        nodemap_[l] = j
        props = Condition.get_node_props(self.graph, j)
        result_.add_node(j, props)
        prev = i
        for n in path:
          if not Condition.has_edge_label(self.graph, prev, j, el):
            # print "No edge", prev, j, "with label", el
            continue
          result_.add_edge(prev, n)
          prev = n
        
        #### Find next neighbors for all candidates
        self.process_neighbors(result_, touched_, nodemap_, unproc_)
    """
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
          sum = 0
          for j in nodes:
            if Condition.satisries_node(self.graph, j, ll, lp):
              sum += log(self.getRWR(j, i) / num)
          log_good += sum / rwrs[l]
      
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
    kl = Condition.get_node_label(self.query, k)  # Label of source
    ll = Condition.get_node_label(self.query, l)  # Label of destination
    
    logging.debug("## Neighbor expander from: " + str(i) + "[" + str(k) + "]")
    max_good = float('-inf')
    candidates_j = Condition.filter_nodes(self.graph, ll, {})
    # candidates_i = [i_ for i_ in self.graph.nodes() if self.get_node_label(self.graph, i_) == kl]
    # candidates_j = [j_ for j_ in self.graph.nodes() if self.get_node_label(self.graph, j_) == ll]
    j = []
    # print "candidates", candidates_j
    # print "result nodes", result.nodes()
    
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
  
  #def bridge_label(self, i, j, label):
  #  return self.extracts[label].getPath(i, j)

  
  
  def computeRWR(self):
    RESTART_PROB = 0.7
    OG_PROB = 0.1
    rw = RWR.RWR(self.graph)
    for m in self.graph.nodes():
      results = rw.run_exp(m, RESTART_PROB, OG_PROB)
      self.graph_rwr[m] = results
      # print m, self.graph_rwr[m]
  
  def getRWR(self, m, n):
    return self.graph_rwr[m][n]
  
  #def getExtract(self):
  #  return self.extract
  
  @staticmethod
  def rwr(g, m, n):  # Random walk with restart m -> n in g
    RESTART_PROB = 0.7
    OG_PROB = 0.1
    rw = RWR.RWR(g)
    results = rw.run_exp(m, RESTART_PROB, OG_PROB)
    logging.debug("RWR: " + str(m) + " -> " + str(n) + " " + str(results))
    return results[n]



