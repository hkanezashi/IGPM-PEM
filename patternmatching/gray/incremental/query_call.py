from configparser import ConfigParser

import json
from networkx.readwrite import json_graph
import sys
import time
import community

sys.path.append(".")

from patternmatching.gray.incremental.gray_incremental import GRayIncremental
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
  print("Input Graph: " + str(numv) + " vertices, " + str(nume) + " edges")
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


def recursive_louvain(graph, min_size):
  def get_louvain(g):
    partition = community.best_partition(g)
    return partition
  
  def create_reverse_partition(pt):
    rev_part = dict()
    for k in pt:
      v = pt[k]
      if not v in rev_part:
        rev_part[v] = list()
      rev_part[v].append(k)
    return rev_part
  
  def __inner_recursive_louvain(g, num_th):
    pt = get_louvain(g)
    rev_part = create_reverse_partition(pt)
    mem_list = list()
    if len(rev_part) == 1:
      return list(rev_part.values())
    for members in rev_part.values():
      cluster = g.subgraph(members)
      if len(members) >= num_th:
        small_members_list = __inner_recursive_louvain(cluster, num_th)
        mem_list.extend(small_members_list)
      else:
        mem_list.append(members)
    return mem_list
  
  members_list = __inner_recursive_louvain(graph, min_size)
  num_groups = len(members_list)
  part = dict()
  for gid in range(num_groups):
    for member in members_list[gid]:
      part[member] = gid
  return part, create_reverse_partition(part)


def get_seeds(graph, affected, min_size):
  part, rev_part = recursive_louvain(graph, min_size)
  nodes = set()
  for n in affected:
    if not n in part:
      continue
    gid = part[n]
    mem = set(rev_part[gid])
    nodes.update(mem)
  return nodes


def run_gray_iterations(graph, query, directed, cond, base_steps, max_steps, time_limit):
  """Repeat incremental G-Ray algorithms

  :return: List of patterns
  """

  ## Extract edge timestamp
  add_edge_timestamps = nx.get_edge_attributes(graph, "add")  # edge, time
  def dictinvert(d):
    inv = {}
    for k, v in d.items():
      keys = inv.setdefault(v, [])
      keys.append(k)
    return inv
  add_timestamp_edges = dictinvert(add_edge_timestamps)  # time, edges

  step_list = sorted(list(add_timestamp_edges.keys()))

  ## Initialize base graph
  print("Initialize base graph")
  init_graph = nx.MultiDiGraph() if directed else nx.MultiGraph()
  start_steps = step_list[0:base_steps]
  for start_step in start_steps:
    init_edges = add_timestamp_edges[start_step]
    init_graph.add_edges_from(init_edges)
  
  nodes = init_graph.nodes()
  subg = nx.subgraph(graph, nodes)
  init_graph.add_nodes_from(subg.nodes(data=True))
  nx.set_edge_attributes(init_graph, "add", 0)
  print("Nodes:%d, Edges:%d" % (init_graph.number_of_nodes(), init_graph.number_of_edges()))

  time_list = list()

  ## Run base G-Ray
  print("Run base G-Ray")
  st = time.time()
  grm = GRayIncremental(graph, init_graph, query, directed, cond, time_limit)
  grm.run_gray()
  results = grm.get_results()
  ed = time.time()
  elapsed = ed - st
  num_patterns = len(results)
  print("Found %d patterns at step %d: %f[s], throughput: %f" % (num_patterns, 0, elapsed, num_patterns / elapsed))
  time_list.append(elapsed)

  ## Run Incremental G-Ray
  print("Run %d step / %d" % (max_steps, len(step_list)))
  st_step = base_steps + 1
  ed_step = st_step + max_steps
  for t in step_list[st_step:ed_step]:
    print("Run incremental G-Ray: %d" % t)
  
    if enable_profile and t == max_steps - 1:
      import cProfile
      pr = cProfile.Profile()
      pr.enable()
  
    add_edges = add_timestamp_edges[t]
    add_nodes = set([e[0] for e in add_edges] + [e[1] for e in add_edges])
    affected_nodes = get_seeds(grm.graph, add_nodes, grm.community_size)
    print("Add edges: %d" % len(add_edges))
    print("Affected nodes: %d" % len(affected_nodes))
    st = time.time()
    grm.run_incremental_gray(add_edges, affected_nodes)
    results = grm.get_results()
    ed = time.time()
  
    elapsed = ed - st
    num_patterns = len(results)
    print("Found %d patterns at step %d: %f[s], throughput: %f" % (num_patterns, t, elapsed, num_patterns/elapsed))
    time_list.append(elapsed)
    sys.stdout.flush()

  print("Total G-Ray time: %f" % sum(time_list))
  print("Average G-Ray time: %f" % (sum(time_list) / len(time_list)))

  results = grm.get_results()
  return results.values()



def run_query(graph_json, query_args, plot_graph=False, show_graph=False, base_steps=100, max_steps=10, time_limit=0.0):
  """Parse pattern matching query command and options and execute incremental G-Ray

  :param graph_json: Graph JSON file
  :param query_args: Query option list
  :param plot_graph: Whether it plots graphs (default is False)
  :param show_graph: Whether it shows graphs (default is False)
  :param max_steps: Number of steps (default is 100)
  :return:
  """
  # try:
  #   import matplotlib.pylab as plt
  # except RuntimeError:
  #   print("Matplotlib cannot be imported.")
  #   plt = None
  #   plot_graph = False
  #   show_graph = False
  plt = None
  plot_graph = False
  show_graph = False
  posg = None
  
  
  print("Graph JSON file: %s" % graph_json)
  print("Query args: %s" % str(query_args))
  print("Plot graph: %s" % str(plot_graph))
  print("Show graph: %s" % str(show_graph))
  print("Number of steps: %d" % max_steps)

  query, cond, directed, groupby, orderby, aggregates = parse_args(query_args)
  graph = load_graph(graph_json)
  

  if plot_graph:
    posg = nx.spring_layout(graph)
    colors = [label_color[v] for k, v in nx.get_node_attributes(graph, LABEL).items()]
    edge_labels = nx.get_edge_attributes(graph, LABEL)
    nx.draw_networkx(graph, posg, arrows=True, node_color=colors, node_size=1000, font_size=24)
    nx.draw_networkx_edge_labels(graph, posg, labels = edge_labels)
    # nx.draw_networkx(query, node_color='c')
    plt.draw()
    plt.savefig("graph.png")
    if show_graph:
      plt.show()
    plt.close()
  
  
  numv = query.number_of_nodes()
  nume = query.number_of_edges()
  print("Query Graph: " + str(numv) + " vertices, " + str(nume) + " edges")
  
  if plot_graph:
    colors = [label_color[v] for k, v in nx.get_node_attributes(query, LABEL).items()]
    posq = nx.spring_layout(query)
    edge_labels = nx.get_edge_attributes(query, LABEL)
    nx.draw_networkx(query, posq, arrows=True, node_color=colors, node_size=1000, font_size=24)
    nx.draw_networkx_edge_labels(query, posq, labels=edge_labels)
    # nx.draw_networkx(query, node_color='c')
    plt.draw()
    plt.savefig("query.png")
    if show_graph:
      plt.show()
    plt.close()


  patterns = run_gray_iterations(graph, query, directed, cond, base_steps, max_steps, time_limit)
  
  if plot_graph:
    # Export pattern graphs to PNG files
    num = 0
    for qresult in patterns:
      result = qresult.get_graph()
      colors = [label_color[v] for k, v in nx.get_node_attributes(graph, LABEL).items() if result.has_node(k)]
      posr = {n: posg[n] for n in result.nodes()}
      nx.draw_networkx(result, posr, arrows=True, node_color=colors, node_size=1000, font_size=24)
      plt.draw()
      plt.savefig("result" + str(num) + ".png")
      if show_graph:
        plt.show()
      plt.close()
      num += 1
  
  
  ## Post-processing (Grouping and Aggregation)
  ## GroupBy
  if groupby:
    # gr = Grouping.Grouping(groupby)
    gr = Grouping.Grouping()
    groups = gr.groupBy(patterns)
    for k, v in groups:
      print(k, len(v))
  
  ## OrderBy
  if orderby:
    od = Ordering.Ordering(orderby)
    ordered = od.orderBy(patterns)
    for result in patterns:
      g = result.get_graph()
      print(g.nodes(), g.edges())
  
  ## Aggregator
  if aggregates:
    for aggregate in aggregates:
      ag = Aggregator(aggregate)
      ret = ag.get_result(patterns)
      print(aggregate, ret)
  
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
  basesteps = int(conf.get("G-Ray", "base_steps"))
  qargs = conf.get("G-Ray", "query").split(" ")
  time_limit = float(conf.get("G-Ray", "time_limit"))
  print("Graph file: %s" % gfile)
  print("Query args: %s" % str(qargs))
  print("Log level: %s" % str(loglevel))
  logging.basicConfig(level=loglevel)
  run_query(gfile, qargs, True, False, basesteps, steps, time_limit)


