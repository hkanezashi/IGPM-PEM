# G-Ray for multiple threads
import networkx as nx
import json
import time
from networkx.readwrite import json_graph
import sys
from ConfigParser import ConfigParser
from multiprocessing import Process  # Use Process instead of Pool

sys.path.append(".")

from patternmatching.query.ConditionParser import ConditionParser
from patternmatching.gray.gray_multiple import GRayMultiple
from patternmatching.query.Condition import *


class GRayParallel(GRayMultiple, object):
  
  def __init__(self, graph, query, directed, cond, time_limit, seeds=None):
    super(GRayParallel, self).__init__(graph, query, directed, cond, time_limit)
    self.seeds = seeds
    self.called = 0
  
  def computeRWR(self):
    self.graph_rwr.rwr_set(self.seeds)
  
  def process_gray(self):
    k = list(self.query.nodes())[0]
    kl = Condition.get_node_label(self.query, k)
    kp = Condition.get_node_props(self.query, k)
    
    if self.seeds is None:
      self.seeds = Condition.filter_nodes(self.graph, kl, kp)
    
    if not self.seeds:
      print("No more seed vertices available. Exit G-Ray algorithm.")
      return
    else:
      print("Number of seeds: %d" % len(self.seeds))

    st = time.time()  # Start time
    for i in self.seeds:
      self.current_seed = i
  
      result = nx.MultiDiGraph() if self.directed else nx.MultiGraph()
      touched = []
      nodemap = {}
      unprocessed = self.query.copy()
  
      il = Condition.get_node_label(self.graph, i)
      props = Condition.get_node_props(self.graph, i)
      nodemap[k] = i
      result.add_node(i)
      result.nodes[i][LABEL] = il
      for name, value in props.iteritems():
        result.nodes[i][name] = value
      touched.append(k)
  
      self.process_neighbors(result, touched, nodemap, unprocessed)
  
      if 0.0 < self.time_limit < time.time() - st:
        print("Timeout G-Ray iterations")
        break
  
  

def parse_query(query_args):
  vsymbols = set()  ## Vertices (symbol)
  esymbols = {}  ## Edges (symbol -> vertex tuple)
  vlabels = {}  ## Vertex Label (symbol -> label)
  elabels = {}  ## Edge Label (symbol -> label)
  epaths = set()  ## Special Edge as Path
  cond = None  ## Complex conditions
  directed = False
  groupby = []  ## GroupBy symbols
  orderby = []  ## OrderBy symbols
  aggregates = []  ## Aggregate Operators
  
  mode = 'command'
  for arg in query_args:
    if arg == '--graph':
      mode = 'graph'
    elif arg == '--vertex':
      mode = 'vertex'
    elif arg == '--edge':
      mode = 'edge'
    elif arg == '--path':
      mode = 'path'
    elif arg == '--vertexlabel':
      mode = 'vlabel'
    elif arg == '--edgelabel':
      mode = 'elabel'
    elif arg == '--condition':
      mode = 'condition'
    elif arg == '--directed':
      directed = True
    elif arg == '--groupby':
      mode = 'groupby'
    elif arg == '--orderby':
      mode = 'orderby'
    elif arg == '--aggregate':
      mode = 'aggregate'
    else:
      if mode == 'graph':
        continue  ## Discard graph name
      elif mode == 'vertex':
        vsymbols.add(arg)
      elif mode == 'edge':
        s = arg.split(":")
        esymbols[s[0]] = (s[1], s[2])
      elif mode == 'path':
        s = arg.split(":")
        esymbols[s[0]] = (s[1], s[2])
        epaths.add(s[0])
      elif mode == 'vlabel':
        s = arg.split(":")
        vlabels[s[0]] = s[1]
      elif mode == 'elabel':
        s = arg.split(":")
        elabels[s[0]] = s[1]
      elif mode == 'condition':
        cond = ConditionParser(arg)
      elif mode == 'groupby':
        groupby.append(arg)
      elif mode == 'orderby':
        orderby.append(arg)
      elif mode == 'aggregate':
        aggregates.append(arg)
  
  if directed:
    query = nx.MultiDiGraph()
  else:
    query = nx.MultiGraph()
  
  for v in vsymbols:
    if v in vlabels:
      query.add_node(v, label=vlabels[v])
    else:
      query.add_node(v)
  
  for e in esymbols:
    edge = esymbols[e]
    if e in elabels:
      query.add_edge(*edge, label=elabels[e])
    else:
      query.add_edge(*edge)
    if e in epaths:
      src, dst = edge
      Condition.set_path(query, src, dst)
  
  return query, cond



def load_graph(graph_json):
  with open(graph_json, "r") as f:
    json_data = json.load(f)
    graph = json_graph.node_link_graph(json_data)
  return graph



# def chunks(l, n):
#   """Divide a list of nodes `l` in `n` chunks"""
#   l_c = iter(l)
#   while 1:
#     x = list(itertools.islice(l_c, n))
#     if not x:
#       return
#     yield x


def split_list(seeds, num_proc):
  num_seeds = len(seeds)
  num_members = num_seeds / num_proc
  seed_lists = list()
  for i in range(num_proc):
    st = i * num_members
    ed = num_seeds if (i == num_proc - 1) else (i + 1) * num_members
    seed_lists.append(seeds[st:ed])
  return seed_lists


def split_list_wcc(g, num_proc):
  seed_lists = {pid: list() for pid in range(num_proc)}
  wccs = [l for l in sorted(nx.weakly_connected_components(nx.DiGraph(g)), key=len, reverse=True)]
  # print len(wccs), sum([len(l) for l in wccs])
  for wcc in wccs:
    pid = min(seed_lists.keys(), key=lambda n:len(seed_lists[n]))
    seed_lists[pid].extend(list(wcc))
  # for k, v in seed_lists.iteritems():
  #   print k, len(v)
  # print len(seed_lists)
  return seed_lists.values()



def run_query_part(args):
  g, q_args, time_limit, seeds, pid = args
  # g = load_graph(g_file)
  query, cond = parse_query(q_args)
  directed = g.is_directed()
  st = time.time()
  grp = GRayParallel(g, query, directed, cond, time_limit, seeds)
  grp.run_gray()
  ed = time.time()
  num_patterns = len(grp.get_results())
  print("G-Ray part %d:%d %f[s]" % (pid, num_patterns, (ed - st)))
  return num_patterns



def run_query_parallel(g_file, q_args, time_limit=0.0, num_proc=1, max_steps=10):
  # directed = query.is_directed()
  g = load_graph(g_file)
  print("Number of vertices: %d" % g.number_of_nodes())
  print("Number of edges: %d" % g.number_of_edges())

  ## Extract edge timestamp
  add_edge_timestamps = nx.get_edge_attributes(g, "add")  # edge, time
  def dictinvert(d):
    inv = {}
    for k, v in d.iteritems():
      keys = inv.setdefault(v, [])
      keys.append(k)
    return inv
  add_timestamp_edges = dictinvert(add_edge_timestamps)  # time, edges
  step_list = sorted(list(add_timestamp_edges.keys()))

  ## Initialize base graph
  print("Initialize base graph")
  start_step = step_list[0]
  init_edges = add_timestamp_edges[start_step]
  init_graph = nx.MultiGraph()
  init_graph.add_edges_from(init_edges)

  nodes = init_graph.nodes()
  subg = nx.subgraph(g, nodes)
  init_graph.add_nodes_from(subg.nodes(data=True))
  nx.set_edge_attributes(init_graph, 0, "add")
  print init_graph.number_of_nodes(), init_graph.number_of_edges()
  
  procs = list()
  node_chunks = split_list(nodes, num_proc) # list(chunks(g.nodes(), int(g.order() / node_divisor)))
  for pid in range(num_proc):
    procs.append(Process(target=run_query_part, args=((g, q_args, time_limit, node_chunks[pid], pid),)))
  for proc in procs:
    proc.start()
  print("Started")
  for proc in procs:
    proc.join()
  print("Finished")

  ## Run Incremental G-Ray
  print("Run %d steps out of %d" % (max_steps, len(step_list)))
  for t in step_list[1:max_steps]:
    print("Run incremental G-Ray: %d" % t)
  
    add_edges = add_timestamp_edges[t]
    print("Add edges: %d" % len(add_edges))
    seeds = set([src for (src, dst, _) in add_edges] + [dst for (src, dst, _) in add_edges])  # Affected nodes

    st = time.time()
    procs = list()
    node_chunks = split_list(seeds, num_proc)
    for pid in range(num_proc):
      procs.append(Process(target=run_query_part, args=((g, q_args, time_limit, node_chunks[pid], pid),)))
    for proc in procs:
      proc.start()
    for proc in procs:
      proc.join()
    ed = time.time()

    elapsed = ed - st
    print("Time at step %d: %f[s]" % (t, elapsed))
  
  
  

if __name__ == "__main__":
  
  argv = sys.argv
  if len(argv) < 2:
    print("Usage: python %s [ConfFile]" % argv[0])
    sys.exit(1)
  
  conf = ConfigParser()
  conf.read(argv[1])
  
  gfile = conf.get("G-Ray", "input_json")
  steps = int(conf.get("G-Ray", "steps"))
  qargs = conf.get("G-Ray", "query").split(" ")
  timelimit = float(conf.get("G-Ray", "time_limit"))
  numproc = int(conf.get("G-Ray", "num_proc"))
  maxsteps = int(conf.get("G-Ray", "steps"))
  print("Graph file: %s" % gfile)
  print("Query args: %s" % str(qargs))
  print("Number of proc: %d" % numproc)
  
  # g = load_graph(gfile)
  # print("Number of vertices: %d" % g.number_of_nodes())
  # print("Number of edges: %d" % g.number_of_edges())
  # q, cond = parse_query(qargs)
  
  run_query_parallel(gfile, qargs, timelimit, numproc, maxsteps)
  
  




