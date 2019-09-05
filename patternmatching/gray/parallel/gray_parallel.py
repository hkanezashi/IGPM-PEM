"""
Extract multiple pattern subgraphs with G-Ray algorithm

Tong, Hanghang, et al. "Fast best-effort pattern matching in large attributed graphs."
Proceedings of the 13th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2007.
"""

import json
from networkx.readwrite import json_graph
from math import log
import time
from functools import partial

from multiprocessing.dummy import Pool
from multiprocessing import Manager
# import pathos.pools as pp

import sys
sys.path.append(".")

from patternmatching.gray.parallel.query_call import parse_args
from patternmatching.gray import rwr, extract
from patternmatching.query.Condition import *
from patternmatching.query import QueryResult


def valid_result(result, query, nodemap):
  ## Note: The number of vertices and edges of graphs with paths will vary
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

  for qn, rn in nodemap.iteritems():
    qd = query.degree(qn)
    rd = result.degree(rn)
    # print "degree:", qn, qd, rn, rd
    if qd != rd:
      return False

  return True


def equal_graphs(g1, g2):
  ns1 = set(g1.nodes())
  ns2 = set(g2.nodes())
  diff = ns1 ^ ns2
  if diff:  ## Not empty (has differences)
    return False
  
  es1 = set(g1.edges())
  es2 = set(g2.edges())
  diff = es1 ^ es2
  if diff:
    return False
  
  return True


def load_graph(graph_json):
  with open(graph_json, "r") as f:
    json_data = json.load(f)
    graph = json_graph.node_link_graph(json_data)
  numv = graph.number_of_nodes()
  nume = graph.number_of_edges()
  print("Input Graph: " + str(numv) + " vertices, " + str(nume) + " edges")
  return graph


def computeRWR(g):
  RESTART_PROB = 0.7
  OG_PROB = 0.1
  rw = rwr.RWR_WCC(g, RESTART_PROB, OG_PROB)
  rw.rwr_all()
  return rw




def run_parallel_gray(gfile, qargs, num_proc):
  """
  
  :param gfile: Graph JSON file
  :param qargs: Query args
  :param num_proc: Number of processes
  :return:
  """
  
  manager = Manager()
  patterns = manager.dict()
  # g_rwr = dict()
  g_pre = dict()
  
  g_ = load_graph(gfile)
  query_, cond_, _, _, _, _ = parse_args(qargs)
  
  st = time.time()
  g_rwr_ = computeRWR(g_)
  # for src, dsts in g_rwr_.iteritems():
  #   d = dict()
  #   for dst, score in dsts.iteritems():
  #     d[dst] = score
  #   g_rwr[src] = d
  ed = time.time()
  print("RWR time: %f" % (ed - st))

  st = time.time()
  g_ext_ = extract.Extract(g_, g_rwr_)
  g_ext_.computeExtract()
  for src, dsts in g_ext_.pre.iteritems():
    d = dict()
    for dst, p in dsts.iteritems():
      d[dst] = p
    g_pre[src] = d
  ed = time.time()
  print("Extract time: %f" % (ed - st))


  # Find query candidates
  q_seed_ = list(query_.nodes())[0]
  kl = Condition.get_node_label(query_, q_seed_)
  kp = Condition.get_node_props(query_, q_seed_)
  seeds = Condition.filter_nodes(g_, kl, kp)  # Find all candidates
  if not seeds:  ## No seed candidates
    print("No more seed vertices available. Exit G-Ray algorithm.")
    return
  
  
  # Split seed list
  num_seeds = len(seeds)
  num_members = num_seeds / num_proc
  seed_lists = list()
  for i in range(num_proc):
    st = i * num_members
    ed = num_seeds if (i == num_proc-1) else (i+1) * num_members
    seed_lists.append(seeds[st:ed])
  
  
  def process_multiple_gray(seed_list, q_seed, q_args, g_rwr, g_pre):
    # st = time.time()
    # g = nx.MultiGraph(g_)
    # g_rwr = dict(g_rwr_)
    # g_pre = dict(g_ext_.pre)
    # ed = time.time()
    # print ed - st
    
    query, cond, _, _, _, _ = parse_args(q_args)
    # g_rwr_ = g_rwr.copy()
    # g_pre_ = g_pre.copy()
    
    for seed in seed_list:
      process_single_gray(seed, q_seed, query, cond, g_rwr, g_pre)
  
  
  def process_single_gray(seed, q_seed, query, cond, g_rwr, g_pre):
    """
    :param q_seed: Seed vertex of query graph
    :param cond: Condition parser
    :param seed: Seed vertex of data graph
    :return:
    """

    def getRWR(i, j):
      if not i in g_rwr:
        return 0.0
      else:
        return g_rwr[i].get(j, 0.0)

    def bridge(i, j):
      lst = list()
      if not i in g_pre:
        return lst
      if not j in g_pre[i]:
        return lst
      v = j
      while v != i:
        lst.append(v)
        if not v in g_pre[i]:
          return []
        v = g_pre[i][v]
      lst.reverse()
      return lst
    
    def neighbor_expander(i, k, l, ret, rev_edge):
      # ll = Condition.get_node_label(query, l)  # Label of destination
      max_good = float('-inf')
      candidates_j = g_rwr.keys() # Condition.filter_nodes(g, ll, {})
      j = []
  
      for j_ in candidates_j:
        if j_ in ret.nodes() or j_ == i:
          continue
    
        if rev_edge:
          log_good = log(getRWR(j_, i) + 1.0e-10)  # avoid math domain errors when the vertex is unreachable
        else:
          log_good = log(getRWR(i, j_) + 1.0e-10)  # avoid math domain errors when the vertex is unreachable
    
        if log_good > max_good:
          j = [j_]
          max_good = log_good
        elif log_good >= max_good - 1.0e-5:  ## Almost same, may be a little smaller due to limited precision
          j.append(j_)
      return j
    
    
    result = nx.Graph()
    touched = []
    nodemap = {}  ## Query Vertex -> Graph Vertex
    unproc = query.copy()
    # il = Condition.get_node_label(g, seed)
    # props = Condition.get_node_props(g, seed)
    nodemap[q_seed] = seed
    result.add_node(seed)
    # result.nodes[seed][LABEL] = il
    # for name, value in props.iteritems():
    #   result.nodes[seed][name] = value
    touched.append(q_seed)
    
    
    ## Process neighbors
    snapshot_stack = list()
    snapshot_stack.append((result, touched, nodemap, unproc))
    
    while snapshot_stack:
      result, touched, nodemap, unproc = snapshot_stack.pop()
      
      if unproc.number_of_edges() == 0:
        if valid_result(result, query, nodemap):
          # append_results(cond, seed, result, nodemap)
          print("Append results: %s" % str(result.nodes()))
        continue
  
      k = None
      l = None
      reversed_edge = False
      for k_ in touched:
        ## Edge
        for l_ in query.neighbors(k_):
          if not unproc.has_edge(k_, l_):
            continue
          l = l_
          break
        if l is not None:
          k = k_
          break
    
      if l is None:  # No more matched vertices
        continue
  
      i = nodemap[k]
      touched.append(l)
  
      #### Find a path or edge (Begin)
      src, dst = (l, k) if reversed_edge else (k, l)
  
      elabel = Condition.get_edge_label(unproc, src, dst)
      if elabel is None:  # Any label is OK
        eid = None
        el = ''
      else:
        eid = elabel[0]
        el = elabel[1]
      Condition.remove_edge_from_id(unproc, src, dst, eid)
  
      # Find a neighbor and connecting edge
      if l in nodemap:
        jlist = [nodemap[l]]
      else:
        # lock.acquire()
        jlist = neighbor_expander(i, k, l, result, reversed_edge)  # find j(l) from i(k)
        # lock.release()
        if not jlist:  ## No more neighbor candidates
          continue
  
      for j in jlist:
        g_src, g_dst = (j, i) if reversed_edge else (i, j)
        if g_src == g_dst:  ## No bridge process necessary
          continue
        # lock.acquire()
        path = bridge(g_src, g_dst)
        # lock.release()
        if not path:
          continue
    
        result_ = nx.MultiGraph(result)
        touched_ = list(touched)
        nodemap_ = dict(nodemap)
        unproc_ = nx.MultiGraph(unproc)
    
        if l in nodemap_ and nodemap_[l] != j:  ## Need to replace mapping
          prevj = nodemap_[l]
          result_.remove_node(prevj)
        nodemap_[l] = j
        result_.add_node(j)
        ## Property addition is not necessary
        # props = Condition.get_node_props(g, j)
        # for k, v in props.iteritems():
        #   result_.nodes[j][k] = v
    
        prev = g_src
        valid = True
        for n in path:
          # if not Condition.has_edge_label(g, prev, n, el):
          #   valid = False
          #   break
          result_.add_edge(prev, n)
          prev = n
        if valid:
          snapshot_stack.append((result_, touched_, nodemap_, unproc_))
  
  
  
  def append_results(cond, seed, result, nodemap):
    if cond is not None and not cond.eval(result, nodemap):
      return False  ## Not satisfied with complex condition

    for r in patterns.values():
      rg = r.get_graph()
      if equal_graphs(rg, result):
        return False
    qresult = QueryResult.QueryResult(result, nodemap)
    patterns[seed] = qresult
    print("Append Results: %s" % str(result.nodes()))
    return True
  
  
  st = time.time()
  pool = Pool(num_proc)
  # pool = pp.ThreadPool(num_proc)
  # pool = pp.ProcessPool(num_proc)  ## Multiprocessing is slow
  
  pool.map_async(partial(process_multiple_gray, q_seed=q_seed_, q_args=list(qargs), g_rwr=g_rwr_, g_pre=g_pre), seed_lists)
  # query, cond, _, _, _, _ = parse_args(qargs)
  # pool.map_async(partial(process_single_gray, q_seed=q_seed_, query=query_, cond=cond_, g_rwr=deepcopy(g_rwr), g_pre=deepcopy(g_pre)), seeds)
  
  pool.close()
  pool.join()
  ed = time.time()
  print("Parallel G-Ray time: %f" % (ed - st))



