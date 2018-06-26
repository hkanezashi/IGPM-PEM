"""
Extract multiple pattern subgraphs with G-Ray algorithm

Tong, Hanghang, et al. "Fast best-effort pattern matching in large attributed graphs."
Proceedings of the 13th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2007.
"""

import time
import numpy as np
import statistics

from patternmatching.gray import extract
from patternmatching.gray.incremental.extract_incremental import Extract
from patternmatching.gray.gray_multiple import GRayMultiple
from patternmatching.query.Condition import *
from patternmatching.query import QueryResult


def equal_graphs(g1, g2):
  ns1 = set(g1.nodes())
  ns2 = set(g2.nodes())
  # logging.debug("Node 1: " + str(ns1))
  # logging.debug("Node 2: " + str(ns2))
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


def get_init_graph(graph):
  edges = graph.edges(data=True)
  init_edges = [e for e in edges if e[2]["add"] == 0]
  init_graph = nx.Graph()
  init_graph.add_edges_from(init_edges)
  # print init_graph.edges.data(data=True)
  return init_graph


def update_added_nodes(nodes, added):
  """Add nodes to the original node list with keeping timestamp order
  https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-whilst-preserving-order
  :param nodes: Original nodes
  :param added: Additional nodes
  :return: Updated nodes
  """
  seen = set(nodes)
  seen_add = seen.add
  new_added = [n for n in added if not (n in seen or seen_add(n))]
  print("Original: %d, Added: %d, Total: %d" % (len(nodes), len(added), len(nodes)+len(new_added)))
  return nodes + new_added


def added_nodes_priority(nodes, added):
  """Sort added node list with high priority for already added one
  Keep order in original existing node list
  Example: added_nodes_priority([1, 2, 3, 4, 7, 8], [2, 5, 1, 6]) --> [1, 2, 5, 6]
  
  :param nodes: Original existing node list
  :param added: Additional node list
  :return: Additional node list with priority (high: already added)
  """
  nodes_set = set(nodes)
  added_set = set(added)
  already_added = [n for n in nodes if n in added_set]
  new_added = [n for n in added if not n in nodes_set]
  assert len(added) == len(already_added) + len(new_added)
  return already_added + new_added


def compute_community_size(g):
  sizes = [len(wcc) for wcc in list(nx.weakly_connected_components(g.to_directed())) if len(wcc) > 1]
  init_size = statistics.median(sizes)
  print("Community size: %d" % init_size)
  return init_size


class GRayIncremental(GRayMultiple, object):
  def __init__(self, orig_graph, graph, query, directed, cond, time_limit):
    super(GRayIncremental, self).__init__(graph, query, directed, cond, time_limit)
    self.elapsed = 0.0  # Elapsed time
    self.nodes = list()  # Added nodes (must be sorted by added timestamp)
    self.orig_graph = orig_graph
    self.community_size = query.number_of_nodes() # compute_community_size(graph)
  
  def update_graph(self, nodes, edges):
    self.graph.add_nodes_from(nodes)
    self.graph.add_edges_from(edges)
  
  
  def run_gray(self):
    """
    Run batch G-Ray algorithm
    :return:
    """
    start = time.time()
    logging.info("---- Start Batch G-Ray ----")
    st = time.time()
    self.computeRWR_batch()
    ed = time.time()
    logging.info("#### Compute RWR: %f [s]" % (ed - st))

    st = time.time()
    ext = Extract(self.graph, self.graph_rwr)
    ext.computeExtract_batch()
    self.extracts[''] = ext
    ed = time.time()
    logging.info("#### Compute Paths: %f [s]" % (ed - st))

    st = time.time()
    self.process_gray()
    ed = time.time()
    logging.info("#### Compute G-Ray: %f [s]" % (ed - st))
    end = time.time()
    self.elapsed = end - start  # set elapsed time


  def process_incremental_gray(self, nodes):
    """
    
    :param nodes: Newly affected nodes for re-computation
    :return:
    """
    logging.debug("#### Find Seeds")
    k = list(self.query.nodes())[0]
    kl = Condition.get_node_label(self.query, k)
    kp = Condition.get_node_props(self.query, k)
    seeds = Condition.filter_nodes(self.graph, kl, kp, nodes)  # Find all candidate seed vertices
    # seeds = set(nodes) & set(seeds)  # Seed candidates are only updated nodes
    print "Number of seeds:", len(seeds)
    
    if not seeds:  ## No seed candidates
      logging.debug("No more seed vertices available. Exit G-Ray algorithm.")
      return

    nodeset = added_nodes_priority(self.nodes, seeds)
  
    st = time.time()  # Start time
    buf = ""
    # for nodeset in [exist_nodes, new_nodes]: # already added vertices have priority
    for i in nodeset:
      logging.debug("#### Choose Seed: " + str(i))
      self.current_seed = i
      result = nx.MultiDiGraph() if self.directed else nx.MultiGraph()
      touched = []
      nodemap = {}  ## Query Vertex -> Graph Vertex
      unprocessed = self.query.copy()
    
      il = Condition.get_node_label(self.graph, i)
      props = Condition.get_node_props(self.graph, i)
      nodemap[k] = i
      result.add_node(i)
      result.nodes[i][LABEL] = il
      for name, value in props.iteritems():
        result.nodes[i][name] = value
    
      touched.append(k)
      
      tmp = self.count
      self.process_neighbors(result, touched, nodemap, unprocessed)
      buf += "*" if (tmp < self.count) else "."  # Found new patterns?
      ts = time.time()
      if 0.0 < self.time_limit < ts - st:
        print(buf)
        print("Timeout G-Ray iterations: %f" % (ts - st))
        return
    print buf
  
  
  def get_observation(self):
    """Extract current G-Ray environment space features (Used for reinforcement learning)
    :return: List of space features
    """
    num_nodes = self.graph.number_of_nodes()
    num_edges = self.graph.number_of_edges()
    return np.array([num_nodes, num_edges])
  
  
  def get_reward(self, max_value):
    """Get reward of reinforcement learning
    :param max_value: Maximum reward value
    :return: Reward value of current step
    """
    reward = len(self.results) / self.elapsed
    # reward = 100.0 / self.elapsed
    print("Patterns: %d, Time: %f, Reward: %f" % (len(self.results), self.elapsed, reward))
    return min(max_value, reward)
  
  
  def add_edges(self, add_edges):
    self.graph.add_edges_from(add_edges)
    
    

  def run_incremental_gray(self, add_edges, affected_nodes=None):
    """
    Run incremental G-Ray algorithm
    :param add_edges: Added edges
    :param affected_nodes: Nodes for recomputation
    :return:
    """
    if affected_nodes is None:
      nodes = set([src for (src, dst, _) in add_edges] + [dst for (src, dst, _) in add_edges])  # Affected nodes
    else:
      nodes = affected_nodes
    # self.graph.add_nodes_from(nodes)
    subg = nx.subgraph(self.orig_graph, nodes)
    self.graph.add_nodes_from(subg.nodes(data=True))
    self.graph.add_edges_from(add_edges)
    
    logging.info("Number of vertices: %d" % len(self.nodes))
    logging.info("Number of edges: %d" % self.graph.number_of_edges())

    logging.info("---- Start Incremental G-Ray ----")
    logging.info("Number of re-computation nodes: %d" % len(nodes))
    
    start = st = time.time()
    self.compute_part_RWR(nodes, add_edges)
    ed = time.time()
    logging.info("#### Compute RWR: %f [s]" % (ed - st))
    
    st = time.time()
    ext = self.extracts['']  # Extract(self.graph, self.graph_rwr)
    ext.computeExtract_incremental(nodes)
    self.extracts[''] = ext
    ed = time.time()
    logging.info("#### Compute Paths: %f [s]" % (ed - st))

    # pr = cProfile.Profile()
    # pr.enable()
    st = time.time()
    self.process_incremental_gray(nodes)
    end = ed = time.time()
    logging.info("#### Compute G-Ray: %f [s]" % (ed - st))
    self.elapsed = end - start
    self.nodes = update_added_nodes(self.nodes, nodes)


  
  def getExtract(self, label):
    if not label in self.extracts:
      ext = extract.Extract(self.graph, self.graph_rwr, label)
      ext.computeExtract()
      self.extracts[label] = ext
    return self.extracts[label]

  def append_results(self, result, nodemap):
    if self.cond is not None and not self.cond.eval(result, nodemap):
      return False  ## Not satisfied with complex condition

    ## Remove duplicates
    for r in self.results.values():
      rg = r.get_graph()
      if equal_graphs(rg, result):
        return False
    
    ## Register the result pattern
    qresult = QueryResult.QueryResult(result, nodemap)
    self.results[self.current_seed] = qresult  # Register QueryResult of current seed
    
    logging.debug("Result nodes:" + str(result.nodes()))
    return True

  
  def process_neighbors(self, result, touched, nodemap, unproc):
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
    kl = None
    reversed_edge = False
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
    
  def compute_part_RWR(self, nodes, edges):
    """Compute incremental RWR
    """
    st = time.time()
    # rw = rwr.RWR(self.graph)
    # rw = rwr.RWR_WCC(self.graph, RESTART_PROB, OG_PROB)
    
    recomp_nodes = added_nodes_priority(self.nodes, nodes)
    self.graph_rwr.add_edges(edges)
    self.graph_rwr.rwr_set(recomp_nodes)
    # for m in recomp_nodes:
      # self.graph_rwr.rwr_single(m)
      # if 0.0 < self.time_limit < time.time() - st:
      #   print("Timeout RWR iterations")
      #   return
  
  
  def separate_exist_nodes(self, affected_nodes):
    """Separate affected nodes into already exist and newly appear
    
    :param affected_nodes:
    :return:
    """
    affected_set = set(affected_nodes)
    exist_nodes = set(self.nodes) & affected_set  # already exist nodes
    new_nodes = affected_set - exist_nodes  # newly added vertices
    return exist_nodes, new_nodes



