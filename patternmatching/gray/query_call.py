
import json
from networkx.readwrite import json_graph
import sys
import time
from ConfigParser import ConfigParser  # Use ConfigParser instead of configparser

sys.path.append(".")

from patternmatching.gray.gray_multiple import GRayMultiple
from patternmatching.query.Condition import *
from patternmatching.query.ConditionParser import ConditionParser
from patternmatching.query import Grouping, Ordering
import aggregator


### Label -> matplotlib color string
label_color = {'cyan': 'c', 'magenta': 'm', 'yellow': 'y', 'white': 'w'}

enable_profile = False


def run_query_step(graph_json, query_args, base_steps=100, max_steps=10, time_limit=0.0):
  
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

  print("Graph JSON file: %s" % graph_json)
  print("Query args: %s" % str(query_args))
  print("Number of steps: %d" % max_steps)
  
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
  
  ## Load JSON graph file
  with open(graph_json, "r") as f:
    json_data = json.load(f)
    graph = json_graph.node_link_graph(json_data)
  
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
  
  numv = graph.number_of_nodes()
  nume = graph.number_of_edges()
  print "Input Graph: " + str(numv) + " vertices, " + str(nume) + " edges"

  numv = query.number_of_nodes()
  nume = query.number_of_edges()
  print "Query Graph: " + str(numv) + " vertices, " + str(nume) + " edges"

  ## Extract edge timestamp
  add_edge_timestamps = nx.get_edge_attributes(graph, "add")  # edge, time
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
  init_graph = nx.MultiDiGraph() if directed else nx.MultiGraph()
  start_steps = step_list[0:base_steps]
  for start_step in start_steps:
    init_edges = add_timestamp_edges[start_step]
    node_ids = set([e[0] for e in init_edges] + [e[1] for e in init_edges])
    init_nodes = [(n, p) for (n, p) in graph.nodes(data=True) if n in node_ids]
    init_graph.add_nodes_from(init_nodes)
    init_graph.add_edges_from(init_edges)

  time_list = list()

  ## Run base G-Ray
  print("Run base G-Ray")
  st = time.time()
  grm = GRayMultiple(init_graph, query, directed, cond, time_limit)
  grm.run_gray()
  results = grm.get_results()
  ed = time.time()
  elapsed = ed - st
  print("Found %d patterns at step %d: %f[s]" % (len(results), 0, elapsed))
  time_list.append(elapsed)

  ## Run Incremental G-Ray
  print("Run %d step / %d" % (max_steps, len(step_list)))
  st_step = base_steps + 1
  ed_step = st_step + max_steps
  for t in step_list[st_step:ed_step]:
    print("Run batch G-Ray: %d" % t)
    add_edges = add_timestamp_edges[t]
    print("Add edges: %d" % len(add_edges))
    node_ids = set([e[0] for e in add_edges] + [e[1] for e in add_edges])
    add_nodes = [(n, p) for (n, p) in graph.nodes(data=True) if n in node_ids]
    init_graph.add_nodes_from(add_nodes)
    print("Affected nodes: %d" % len(add_nodes))

    init_graph.add_edges_from(add_edges)
    grm = GRayMultiple(init_graph, query, directed, cond, time_limit)
    numv = init_graph.number_of_nodes()
    nume = init_graph.number_of_edges()
    print "Input Graph: " + str(numv) + " vertices, " + str(nume) + " edges"
    
    st = time.time()
    grm.run_gray()
    results = grm.get_results()
    ed = time.time()
    elapsed = ed - st
    print("Found %d patterns at step %d: %f[s]" % (len(results), t, elapsed))
    time_list.append(elapsed)
    sys.stdout.flush()
  
  print("Total G-Ray time: %f" % sum(time_list))
  print("Average G-Ray time: %f" % (sum(time_list) / len(time_list)))
  


def run_query(graph_json, query_args, plot_graph=False, show_graph=False, time_limit=0.0):
  
  
  """
  ## InputGraph (args[1]): edge list file name
  # graph = nx.read_edgelist(args[1], data=[('label', str)])
  graph = nx.MultiDiGraph(nx.read_gml(gfile))
  """
  try:
    import matplotlib.pylab as plt
  except RuntimeError:
    print("Matplotlib cannot be imported.")
    plt = None
    plot_graph = False
    show_graph = False
  
  ## Query (args[2:]): query graph
  vsymbols = set()  ## Vertices (symbol)
  esymbols = {}  ## Edges (symbol -> vertex tuple)
  vlabels = {}  ## Vertex Label (symbol -> label)
  elabels = {}  ## Edge Label (symbol -> label)
  epaths = set() ## Special Edge as Path
  cond = None ## Complex conditions
  directed = False
  groupby = [] ## GroupBy symbols
  orderby = [] ## OrderBy symbols
  aggregates = [] ## Aggregate Operators
  
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
      
  
  ## Load JSON graph file
  with open(graph_json, "r") as f:
    json_data = json.load(f)
    graph = json_graph.node_link_graph(json_data)
  
  if directed:
    # graph = nx.MultiDiGraph(nx.read_gml(gfile))
    query = nx.MultiDiGraph()
  else:
    # graph = nx.MultiGraph(nx.read_gml(gfile))
    query = nx.MultiGraph()
    # graph.to_undirected()
  
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
  
  numv = graph.number_of_nodes()
  nume = graph.number_of_edges()
  print "Input Graph: " + str(numv) + " vertices, " + str(nume) + " edges"
  # print graph.nodes()
  # print graph.edges()
  
  posg = nx.spring_layout(graph)
  if plot_graph:
    colors = [label_color[v] for k, v in nx.get_node_attributes(graph, LABEL).iteritems()]
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
  print "Query Graph: " + str(numv) + " vertices, " + str(nume) + " edges"
  # print query.nodes()
  # print query.edges()
  
  if plot_graph:
    colors = [label_color[v] for k, v in nx.get_node_attributes(query, LABEL).iteritems()]
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
  
  
  ## Run G-Ray
  st = time.time()

  if enable_profile:
    import cProfile
    pr = cProfile.Profile()
    pr.enable()
  else:
    pr = None
  
  grm = GRayMultiple(graph, query, directed, cond, time_limit)
  grm.run_gray()
  results = grm.get_results()

  if enable_profile:
    pr.disable()
    import pstats
    stats = pstats.Stats(pr)
    stats.sort_stats("tottime")
    stats.print_stats()
  
  ed = time.time()
  print "Found " + str(len(results)) + " patterns."
  print "Elapsed time [s]: " + str(ed - st)
  # print "Extract: " + str(grm.getExtract())
  
  # for pattern in results.values():
  #   print pattern.get_graph().edges()
  
  
  if plot_graph:
    # Export pattern graphs to PNG files
    num = 0
    for qresult in results.values():
      result = qresult.get_graph()
      colors = [label_color[v] for k, v in nx.get_node_attributes(graph, LABEL).iteritems() if result.has_node(k)]
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
    groups = gr.groupBy(results)
    for k, v in groups:
      print k, len(v)
  
  ## OrderBy
  if orderby:
    od = Ordering.Ordering(orderby)
    ordered = od.orderBy(results)
    for result in results:
      g = result.get_graph()
      print g.nodes(), g.edges()
  
  ## Aggregator
  if aggregates:
    for aggregate in aggregates:
      ag = aggregator.Aggregator(aggregate)
      ret = ag.get_result(results)
      print aggregate, ret
  
  return results



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
  run_query_step(gfile, qargs, basesteps, steps, time_limit)


