import networkx as nx
import json
from networkx.readwrite import json_graph
import sys
import time

import gray_incremental
from patternmatching.query.Condition import *
from patternmatching.query.ConditionParser import ConditionParser
from patternmatching.query import Grouping, Ordering
from patternmatching.gray.aggregator import Aggregator


### Label -> matplotlib color string
label_color = {'cyan': 'c', 'magenta': 'm', 'yellow': 'y', 'white': 'w'}
  

def run_query(graph_json, query_args, plot_graph=False, show_graph=False, max_steps=100):
  """Parse pattern matching query command and options and execute incremental G-Ray

  :param graph_json: Graph JSON file
  :param query_args: Query option list
  :param plot_graph: Whether it plots graphs (default is False)
  :param show_graph: Whether it shows graphs (default is False)
  :param max_steps: Number of steps (default is 100)
  :return:
  """
  try:
    import matplotlib.pylab as plt
  except RuntimeError:
    print("Matplotlib cannot be imported.")
    plt = None
    plot_graph = False
    show_graph = False
  
  
  print("Graph JSON file: %s" % graph_json)
  print("Query args: %s" % str(query_args))
  print("Plot graph: %s" % str(plot_graph))
  print("Show graph: %s" % str(show_graph))
  print("Number of steps: %d" % max_steps)


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


  ## Extract edge timestamp
  add_edge_timestamps = nx.get_edge_attributes(graph, "add")  # edge, time

  def dictinvert(d):
    inv = {}
    for k, v in d.iteritems():
      keys = inv.setdefault(v, [])
      keys.append(k)
    return inv

  add_timestamp_edges = dictinvert(add_edge_timestamps)  # time, edges
  # print add_timestamp_edges


  ## Initialize base graph
  print("Initialize base graph")
  init_edges = add_timestamp_edges[0]
  # init_nodes = set([src for (src, dst) in init_edges] + [dst for (src, dst) in init_edges])
  init_graph = nx.MultiDiGraph() if directed else nx.MultiGraph()
  init_graph.add_nodes_from(graph.nodes(data=True))
  # total_edges = init_graph.edges()
  # print total_edges
  # init_graph.remove_edges_from(total_edges)
  init_graph.add_edges_from(init_edges)

  ## Run base G-Ray
  print("Run base G-Ray")
  st = time.time()
  grm = gray_incremental.GRayIncremental(init_graph, query, directed, cond)
  grm.run_gray()
  results = grm.get_results()
  ed = time.time()
  print("Found %d patterns at time %d: %f[s]" % (len(results), 0, (ed - st)))

  ## Run Incremental G-Ray
  for t in range(1, max_steps):
    print("Run incremental G-Ray: %d" % t)
    st = time.time()
    add_edges = add_timestamp_edges[t]
    grm.run_incremental_gray(add_edges)
    results = grm.get_results()
    ed = time.time()
    print("Found %d patterns at time %d: %f[s]" % (len(results), t, (ed - st)))
    # print "Extract: " + str(grm.getExtract())


  results = grm.get_results()
  if plot_graph:
    # Export pattern graphs to PNG files
    num = 0
    for qresult in results:
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
      ag = Aggregator(aggregate)
      ret = ag.get_result(results)
      print aggregate, ret
  
  return results



if __name__ == '__main__':
  args = sys.argv
  if len(args) < 2:
    print "Usage: %s [GraphJSON] [QueryArgs...]" % args[0]
    sys.exit(1)
  gfile = args[1]
  qargs = args[2:]
  print gfile
  print qargs
  logging.basicConfig(level=logging.DEBUG)
  run_query(gfile, qargs, True, False, 10)


