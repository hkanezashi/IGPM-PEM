from ConfigParser import ConfigParser  # Use ConfigParser instead of configparser

import networkx as nx
import json
from networkx.readwrite import json_graph
import sys
import time


sys.path.append(".")
sys.setrecursionlimit(1000)

# from patternmatching.gray.parallel.gray_parallel import GRayParallel
from patternmatching.query.Condition import *
from patternmatching.query.ConditionParser import ConditionParser
from patternmatching.query import Grouping, Ordering
from patternmatching.gray.aggregator import Aggregator


### Label -> matplotlib color string
label_color = {'cyan': 'c', 'magenta': 'm', 'yellow': 'y', 'white': 'w'}

enable_profile = False


def load_graph(graph_json):
  ## Load JSON graph file
  with open(graph_json, "r") as f:
    json_data = json.load(f)
    graph = json_graph.node_link_graph(json_data)
  
  numv = graph.number_of_nodes()
  nume = graph.number_of_edges()
  print "Input Graph: " + str(numv) + " vertices, " + str(nume) + " edges"
  # print graph.nodes()
  # print graph.edges()
  return graph


def parse_args(query_args):
  """Parse query arguments and create query graph object
  :param query_args: Query option list
  :return: Query graph, condition parser, directed flag, groupby symbols, orderby symbols and aggregate operators
  """
  ## Query (args[2:]): query graph
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

  logging.info("Query Arguments: " + " ".join(query_args))

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

  return query, cond, directed, groupby, orderby, aggregates


def run_gray_iterations(graph, query, directed, cond, max_steps, num_proc):
  """Repeat incremental G-Ray algorithms

  :return: List of patterns
  """

  ## Extract edge timestamp
  add_edge_timestamps = nx.get_edge_attributes(graph, "add")  # edge, time
  def dictinvert(d):
    inv = {}
    for k, v in d.iteritems():
      keys = inv.setdefault(v, [])
      keys.append(k)
    return inv
  add_timestamp_edges = dictinvert(add_edge_timestamps)  # time, edges

  ## Initialize base graph
  print("Initialize base graph")
  init_edges = add_timestamp_edges[0]
  init_graph = nx.MultiDiGraph() if directed else nx.MultiGraph()
  init_graph.add_nodes_from(graph.nodes(data=True))
  init_graph.add_edges_from(init_edges)
  nx.set_edge_attributes(init_graph, 0, "add")

  time_list = list()

  ## Run base G-Ray
  print("Run base G-Ray")
  st = time.time()
  grm = GRayParallel(init_graph, query, directed, cond)
  grm.run_gray(num_proc)
  results = grm.get_results()
  ed = time.time()
  elapsed = ed - st
  num_patterns = len(results)
  print("Found %d patterns at step %d: %f[s], throughput: %f" % (num_patterns, 0, elapsed, num_patterns / elapsed))
  time_list.append(elapsed)

  ## Run Incremental G-Ray
  for t in range(1, max_steps):
    print("Run incremental G-Ray: %d" % t)
  
    if enable_profile and t == max_steps - 1:
      import cProfile
      pr = cProfile.Profile()
      pr.enable()
  
    add_edges = add_timestamp_edges[t]
    print("Add edges: %d" % len(add_edges))
    st = time.time()
    grm.run_incremental_gray(add_edges)
    results = grm.get_results()
    ed = time.time()
  
    if enable_profile and t == max_steps - 1:
      pr.disable()
      import pstats
      stats = pstats.Stats(pr)
      stats.sort_stats("tottime")
      stats.print_stats()
  
    elapsed = ed - st
    num_patterns = len(results)
    print("Found %d patterns at step %d: %f[s], throughput: %f" % (num_patterns, t, elapsed, num_patterns/elapsed))
    time_list.append(elapsed)

  print("Total G-Ray time: %f" % sum(time_list))
  print("Average G-Ray time: %f" % (sum(time_list) / len(time_list)))

  results = grm.get_results()
  return results.values()



def run_query(graph_json, query_args, max_steps=100, num_proc=4):
  """Parse pattern matching query command and options and execute incremental G-Ray
  """
  
  print("Graph JSON file: %s" % graph_json)
  print("Query args: %s" % str(query_args))
  print("Number of steps: %d" % max_steps)

  query, cond, directed, groupby, orderby, aggregates = parse_args(query_args)
  graph = load_graph(graph_json)
  
  numv = query.number_of_nodes()
  nume = query.number_of_edges()
  print "Query Graph: " + str(numv) + " vertices, " + str(nume) + " edges"


  patterns = run_gray_iterations(graph, query, directed, cond, max_steps, num_proc)
  
  
  ## Post-processing (Grouping and Aggregation)
  ## GroupBy
  if groupby:
    # gr = Grouping.Grouping(groupby)
    gr = Grouping.Grouping()
    groups = gr.groupBy(patterns)
    for k, v in groups:
      print k, len(v)
  
  ## OrderBy
  if orderby:
    od = Ordering.Ordering(orderby)
    ordered = od.orderBy(patterns)
    for result in patterns:
      g = result.get_graph()
      print g.nodes(), g.edges()
  
  ## Aggregator
  if aggregates:
    for aggregate in aggregates:
      ag = Aggregator(aggregate)
      ret = ag.get_result(patterns)
      print aggregate, ret
  
  return patterns



if __name__ == '__main__':
  args = sys.argv
  if len(args) < 2:
    print("Usage: python %s [ConfFile]" % args[0])
    sys.exit(1)
    
  conf = ConfigParser()
  conf.read(args[1])

  if conf.get("Log", "profile").lower() == "true":
    enable_profile = True
    
  loglevel_str = conf.get("Log", "level").lower()
  if loglevel_str == "debug":
    loglevel = logging.DEBUG
  elif loglevel_str == "info":
    loglevel = logging.INFO
  else:
    loglevel = logging.WARNING

  gfile = conf.get("G-Ray", "input_json")
  steps = int(conf.get("G-Ray", "steps"))
  qargs = conf.get("G-Ray", "query").split(" ")
  time_limit = float(conf.get("G-Ray", "time_limit"))
  num_proc = int(conf.get("G-Ray", "num_proc"))
  print("Graph file: %s" % gfile)
  print("Query args: %s" % str(qargs))
  print("Log level: %s" % str(loglevel))
  logging.basicConfig(level=loglevel)
  run_query(gfile, qargs, steps, num_proc)


