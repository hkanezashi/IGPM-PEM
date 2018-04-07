import networkx as nx
import csv
from networkx.readwrite import json_graph
import json

def load_edgelist_time(in_fname, out_fname):
  """Load edgelist file and output NetworkX graph with generated timestamp as JSON format
  Usage:
  > from patternmatching.gray.incremental.load_edgelist import load_edgelist_time
  > load_edgelist_time("data/Congress/edges", "data/Congress.json")
  
  :param in_fname: Edgelist file name
  :param out_fname: JSON file name
  :return:
  """
  tm_size = 1000
  graph = nx.Graph()
  
  rf = open(in_fname, "r")
  reader = csv.reader(rf, delimiter=" ")
  count = 0
  t = 0
  
  for row in reader:
    src = int(row[0])
    dst = int(row[1])
    
    graph.add_edge(src, dst, label="yes", add=t)
    
    count += 1
    if count % tm_size == 0:
      t += 1  # next timestamp
      
  rf.close()
  
  nx.set_node_attributes(graph, "cyan", "label")
  
  with open(out_fname, "w") as wf:
    data = json_graph.node_link_data(graph)
    json.dump(data, wf, indent=2)
  


def filter_time(in_json, out_json, limit):
  """
  Usage:
  > from patternmatching.gray.incremental.load_edgelist import filter_time
  > filter_time("data/Congress.json", "data/Congress1.json", 10)
  
  :param in_json: Input graph JSON
  :param out_json: Output graph JSON
  :param limit: Upper limit of steps
  :return:
  """
  
  with open(in_json, "r") as rf:
    data = json.load(rf)
    graph1 = json_graph.node_link_graph(data)
  
  # print type(graph1)
  edges = [e for e in graph1.edges(data=True) if e[2]["add"] < limit]
  # print edges[0]
  graph2 = nx.Graph()
  
  graph2.add_edges_from(edges)
  nx.set_node_attributes(graph2, "cyan", "label")
  
  with open(out_json, "w") as wf:
    data = json_graph.node_link_data(graph2)
    json.dump(data, wf, indent=2)
  
  print sum(nx.triangles(graph2).values()) / 3
  # nx.write_gexf(graph2, "output.gexf")


