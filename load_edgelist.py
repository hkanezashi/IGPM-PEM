import networkx as nx
import csv
from networkx.readwrite import json_graph
import json

def load_edgelist(in_fname, out_fname, tm_size):
  """Load edgelist file and output NetworkX graph with generated timestamp as JSON format
  Usage:
  > from patternmatching.gray.incremental.load_edgelist import load_edgelist_time
  > load_edgelist_time("data/Congress/edges", "data/Congress.json", 100)
  > load_edgelist_time("data/imdb/edges", "data/IMDb.json", 100)
  
  :param in_fname: Edgelist file name
  :param out_fname: JSON file name
  :param tm_size: Number of edges per time
  :return:
  """
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
  


def filter_time(in_json, out_json, limit_tm):
  """
  Usage:
  > from patternmatching.gray.incremental.load_edgelist import filter_time
  > filter_time("data/Congress.json", "data/Congress1.json", 10)
  > filter_time("data/IMDb.json", "data/IMDb1.json", 10)
  
  :param in_json: Input graph JSON
  :param out_json: Output graph JSON
  :param limit_tm: Upper limit of steps
  :return:
  """
  
  with open(in_json, "r") as rf:
    data = json.load(rf)
    graph1 = json_graph.node_link_graph(data)
  
  # print type(graph1)
  edges = [e for e in graph1.edges(data=True) if e[2]["add"] < limit_tm]
  print("Number of edges: %d" % len(edges))
  # print edges[0]
  graph2 = nx.Graph()
  
  graph2.add_edges_from(edges)
  nx.set_node_attributes(graph2, "cyan", "label")
  print("Number of nodes: %d" % graph2.number_of_nodes())
  
  with open(out_json, "w") as wf:
    data = json_graph.node_link_data(graph2)
    json.dump(data, wf, indent=2)
  
  print("Triangles: %d" % (sum(nx.triangles(graph2).values()) / 3))
  # nx.write_gexf(graph2, "output.gexf")


def load_edgelist_time(in_fname, out_fname, limit_tm=None):
  graph = nx.Graph()
  
  rf = open(in_fname, "r")
  reader = csv.reader(rf, delimiter=" ")
  
  base_days = None
  
  def sec_to_days(sec):
    return sec / (60 * 60 * 24)
  
  for row in reader:
    src = int(row[0])
    dst = int(row[1])
    tm = int(row[2])
    t = sec_to_days(tm)
    if base_days is None:
      base_days = t

    t -= base_days
    if limit_tm is not None and t > limit_tm:
      break
    graph.add_edge(src, dst, label="yes", add=t)
  
  rf.close()
  
  nx.set_node_attributes(graph, "cyan", "label")
  
  with open(out_fname, "w") as wf:
    data = json_graph.node_link_data(graph)
    json.dump(data, wf, indent=2)





if __name__ == "__main__":
  import sys
  argv = sys.argv
  if len(argv) < 3:
    print("Usage: python %s [tm_size] [limit_tm]" % argv[0])
    exit(1)
  tm_size = int(argv[1])
  limit_tm = int(argv[2])

  print("Convert Congress Data")
  load_edgelist("data/Congress/edges", "data/Congress.json", tm_size)
  filter_time("data/Congress.json", "data/Congress1.json", limit_tm)

  print("Convert IMDb Data")
  load_edgelist("data/imdb/edges", "data/IMDb.json", tm_size)
  filter_time("data/IMDb.json", "data/IMDb1.json", limit_tm)
  
  print("Convert Amazon Data")
  load_edgelist("data/Amazon/edges", "data/Amazon.json", tm_size)
  filter_time("data/Amazon.json", "data/Amazon1.json", limit_tm)
  
  # print("Convert Stackoverflow Data")
  # load_edgelist_time("data/real/sx-stackoverflow.txt", "data/stackoverflow.json", limit_tm)
  # filter_time("data/stackoverflow.json", "data/stackoverflow1.json", limit_tm)
  
