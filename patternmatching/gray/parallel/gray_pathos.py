import csv
import sys
from configparser import ConfigParser
import json
import networkx as nx
from networkx.readwrite import json_graph
import time
from math import log
from pathos.parallel import ParallelPool as Pool
from functools import partial

sys.path.append(".")
from patternmatching.gray.parallel.query_call import parse_args
from patternmatching.gray import rwr, extract
from patternmatching.query.Condition import *


# https://stackoverflow.com/questions/26876898/python-multiprocessing-with-distributed-cluster/26948258
def sleepy_squared(x):
  from time import sleep
  sleep(0.5)
  return x**2

p = Pool(4)
res = p.amap(sleepy_squared, range(10))
print(res.get())

################

port = "5000"


def load_graph(graph_json):
  with open(graph_json, "r") as f:
    json_data = json.load(f)
    graph = json_graph.node_link_graph(json_data)
  numv = graph.number_of_nodes()
  nume = graph.number_of_edges()
  print("Input Graph: " + str(numv) + " vertices, " + str(nume) + " edges")
  return graph


def computeRWR(g):
  graph_rwr = dict()
  RESTART_PROB = 0.7
  OG_PROB = 0.1
  rw = rwr.RWR(g)
  for m in g.nodes():
    results = rw.run_exp(m, RESTART_PROB, OG_PROB)
    graph_rwr[m] = results
  return graph_rwr


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


################

def process_multiple_gray(seed_list, g_file, q_seed, q_args):
  query, cond, _, _, _, _ = parse_args(q_args)
  
  st = time.time()
  g = load_graph(g_file)
  ed = time.time()
  print("Load graph: %f" % (ed - st))

  st = time.time()
  g_rwr = computeRWR(g)
  ed = time.time()
  print("RWR time: %f" % (ed - st))

  st = time.time()
  g_ext = extract.Extract(g, g_rwr)
  g_ext.computeExtract()
  ed = time.time()
  print("Extract time: %f" % (ed - st))
  
  
  for seed in seed_list:
    process_single_gray(seed, q_seed, query, cond, g_rwr, g_ext.pre)
  return len(seed_list)


def process_single_gray(seed, q_seed, query, cond, g_rwr, g_pre):

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
    max_good = float('-inf')
    candidates_j = g_rwr.keys()  # Condition.filter_nodes(g, ll, {})
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



################



def run_parallel_gray(gfile, qargs, hosts):
  
  g_ = load_graph(gfile)
  query_, cond_, _, _, _, _ = parse_args(qargs)
  
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
  num_hosts = len(hosts)
  num_members = num_seeds / num_hosts
  seed_lists = list()
  for i in range(num_hosts):
    st = i * num_members
    ed = num_seeds if (i == num_hosts - 1) else (i + 1) * num_members
    seed_lists.append(seeds[st:ed])

  servers = tuple([":".join([addr, port]) for addr in hosts])
  
  st = time.time()
  pool = Pool(1, servers=servers)
  ret = pool.amap(partial(process_multiple_gray, g_file=gfile, q_seed=q_seed_, q_args=qargs), seed_lists)
  print(ret.get())
  
  pool.close()
  pool.join()
  ed = time.time()
  print("Parallel G-Ray time: %f" % (ed - st))





if __name__ == "__main__":
  argv = sys.argv
  if len(argv) < 3:
    print("Usage: python %s [ConfFile] [PE_HOSTFILE]" % argv[0])
    exit(1)

  conf = ConfigParser()
  conf.read(argv[1])

  gfile = conf.get("G-Ray", "input_json")
  steps = int(conf.get("G-Ray", "steps"))
  qargs = conf.get("G-Ray", "query").split(" ")
  time_limit = float(conf.get("G-Ray", "time_limit"))
  num_proc = int(conf.get("G-Ray", "num_proc"))
  print("Graph file: %s" % gfile)
  print("Query args: %s" % str(qargs))
  print("Number of procs: %d" % num_proc)
  
  with open(argv[2], "r") as rf:
    reader = csv.reader(rf)
    hosts = [h[0] for h in list(reader)]
  
  print("Number of hosts: %d" % len(hosts))
  print("Host list: %s" % str(hosts))
  
  run_parallel_gray(gfile, qargs, hosts)

