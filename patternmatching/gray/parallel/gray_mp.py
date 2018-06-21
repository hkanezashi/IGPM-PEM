# G-Ray for multiple threads
import itertools
import json
import time
from networkx.readwrite import json_graph
import sys
from configparser import ConfigParser
from multiprocessing import Pool, Process

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



def chunks(l, n):
  """Divide a list of nodes `l` in `n` chunks"""
  l_c = iter(l)
  while 1:
    x = list(itertools.islice(l_c, n))
    if not x:
      return
    yield x



def run_query_part(args):
  g_file, q_args, time_limit, seeds, pid = args
  g = load_graph(g_file)
  query, cond = parse_query(q_args)
  directed = g.is_directed()
  st = time.time()
  grp = GRayParallel(g, query, directed, cond, time_limit, seeds)
  grp.run_gray()
  ed = time.time()
  num_patterns = len(grp.get_results())
  print("G-Ray part %d:%d %f[s]" % (pid, num_patterns, (ed - st)))
  return num_patterns



def run_query_parallel(g_file, q_args, time_limit=0.0, num_proc=1):
  # directed = query.is_directed()
  g = load_graph(g_file)
  print("Number of vertices: %d" % g.number_of_nodes())
  print("Number of edges: %d" % g.number_of_edges())
  
  procs = list()
  node_divisor = num_proc
  node_chunks = list(chunks(g.nodes(), int(g.order() / node_divisor)))
  for pid in range(node_divisor):
    procs.append(Process(target=run_query_part, args=((g_file, q_args, time_limit, node_chunks[pid], pid),)))
  for proc in procs:
    proc.start()
  print("Started")
  for proc in procs:
    proc.join()
  print("Finished")
  
  # p = Pool(processes=num_proc)
  # node_divisor = len(p._pool) * 2
  # node_chunks = list(chunks(g.nodes(), int(g.order() / node_divisor)))
  # num_chunks = len(node_chunks)
  # pattern_num = p.map(run_query_part, zip([g_file]*num_chunks, [q_args]*num_chunks,
  #                                         [time_limit]*num_chunks, node_chunks, list(range(num_chunks))))
  # p.close()
  # p.join()
  # print(pattern_num)
  
  

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
  print("Graph file: %s" % gfile)
  print("Query args: %s" % str(qargs))
  print("Number of proc: %d" % numproc)
  
  # g = load_graph(gfile)
  # print("Number of vertices: %d" % g.number_of_nodes())
  # print("Number of edges: %d" % g.number_of_edges())
  # q, cond = parse_query(qargs)
  
  run_query_parallel(gfile, qargs, timelimit, numproc)
  
  




