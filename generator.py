import csv
import json
import sys
import random
from collections import defaultdict
import networkx as nx
from networkx.readwrite import json_graph


random.seed(0)

def _random_subset(seq, m):
  targets = set()
  while len(targets) < m:
    x = random.choice(seq)
    targets.add(x)
  return targets
  

def barabasi_albert_edgelist(n, m):
  """
  https://networkx.github.io/documentation/networkx-1.10/_modules/networkx/generators/random_graphs.html#barabasi_albert_graph
  :param n:
  :param m:
  :return:
  """
  edges = list()
  targets = list(range(m))
  repeated_nodes = []
  source = m
  while source < n:
    edges.extend(zip([source]*m, targets))
    repeated_nodes.extend(targets)
    repeated_nodes.extend([source] * m)
    targets = _random_subset(repeated_nodes, m)
    source += 1
  return edges


def powerlaw_cluster_edgelist(n, m):
  """
  https://networkx.github.io/documentation/networkx-1.10/_modules/networkx/generators/random_graphs.html#powerlaw_cluster_graph
  :param n:
  :param m:
  :return:
  """
  p = 0.2
  edges = list()
  G = nx.empty_graph(m)
  repeated_nodes = list(G.nodes())
  source = m
  while source < n:  # Now add the other n-1 nodes
    possible_targets = _random_subset(repeated_nodes, m)
    target = possible_targets.pop()
    
    G.add_edge(source, target)
    edges.append((source, target))
    
    repeated_nodes.append(target)  # add one node to list for each new link
    count = 1
    while count < m:  # add m-1 more new links
      if random.random() < p:  # clustering step: add triangle
        neighborhood = [nbr for nbr in G.neighbors(target) if not G.has_edge(source, nbr) and not nbr == source]
        if neighborhood:  # if there is a neighbor without a link
          nbr = random.choice(neighborhood)
          
          G.add_edge(source, nbr)
          edges.append((source, nbr)) # add triangle
          
          repeated_nodes.append(nbr)
          count = count + 1
          continue  # go to top of while loop
      target = possible_targets.pop()
      
      G.add_edge(source, target)
      edges.append((source, target))
      
      repeated_nodes.append(target)
      count = count + 1
  
    repeated_nodes.extend([source] * m)  # add source node to list m times
    source += 1
  return edges


def random_regular_edgelist(n, d):
  """
  https://networkx.github.io/documentation/networkx-1.10/_modules/networkx/generators/random_graphs.html#random_regular_graph
  :param n:
  :param d:
  :return:
  """
  def _suitable(edges, potential_edges):
    if not potential_edges:
      return True
    for s1 in potential_edges:
      for s2 in potential_edges:
        if s1 == s2:
          break
        if s1 > s2:
          s1, s2 = s2, s1
        if (s1, s2) not in edges:
          return True
    return False

  def _try_creation():
    edge_set = set()
    edge_list = list()
    stubs = list(range(n)) * d
    while stubs:
      potential_edges = defaultdict(lambda: 0)
      random.shuffle(stubs)
      stubiter = iter(stubs)
      for s1, s2 in zip(stubiter, stubiter):
        if s1 > s2:
          s1, s2 = s2, s1
        if s1 != s2 and ((s1, s2) not in edge_set):
          edge = (s1, s2)
          edge_set.add(edge)
          edge_list.append(edge)
        else:
          potential_edges[s1] += 1
          potential_edges[s2] += 1
      if not _suitable(edge_set, potential_edges):
        return None  # failed to find suitable edge set
      stubs = [node for node, potential in potential_edges.items()
               for _ in range(potential)]
    return edge_list
  
  edges = _try_creation()
  while edges is None:
    edges = _try_creation()
  return edges



def write_edgelist(edges, out_name):
  with open(out_name, "w") as wf:
    writer = csv.writer(wf, delimiter=" ")
    for e in edges:
      writer.writerow(e)


def write_json(edges, out_name, num_inc=100):
  g = nx.MultiGraph()
  for idx, e in enumerate(edges):
    t = idx / num_inc
    g.add_edge(e[0], e[1], label="yes", add=t)
  data = json_graph.node_link_data(g)
  
  with open(out_name, "w") as wf:
    json.dump(data, wf, indent=2)
  



if __name__ == "__main__":

  argv = sys.argv
  if len(argv) < 4:
    print("Usage: python %s [TotalVertices] [EdgeFactor] [VerticesPerStep]" % argv[0])
    exit(1)

  num_v = int(argv[1])
  factor = int(argv[2])
  _num_inc = int(argv[3])
  
  edge_list = barabasi_albert_edgelist(num_v, factor)
  name = "barabasi_albert-%d_%d.json" % (num_v, factor)
  print("Generated %s" % name)
  write_json(edge_list, name, _num_inc)

  edge_list = powerlaw_cluster_edgelist(num_v, factor)
  name = "powerlaw_cluster-%d_%d.json" % (num_v, factor)
  print("Generated %s" % name)
  write_json(edge_list, name, _num_inc)

  edge_list = random_regular_edgelist(num_v, factor)
  name = "random_regular-%d_%d.json" % (num_v, factor)
  print("Generated %s" % name)
  write_json(edge_list, name, _num_inc)


