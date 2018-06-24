import csv
import json
import networkx as nx
from networkx.readwrite import json_graph
import sys


argv = sys.argv
if len(argv) < 3:
  print("Usage: python %s [InputCSV] [OutputJSON]")
  exit(1)

input_csv = argv[1]
output_json = argv[2]

g = nx.MultiGraph()

count = 0

with open(input_csv, "r") as rf:
  reader = csv.reader(rf, delimiter=",")
  for row in reader:
    src = int(row[0])
    dst = int(row[1])
    ts = int(row[2]) / (24 * 60 * 60)
    g.add_edge(src, dst, t=ts)
    count += 1
    if count % 10000 == 0:
      print count

tss = nx.get_edge_attributes(g, "t")
base_step = min(tss.values())
print("Base step: %d" % base_step)
new_tss = {k: v - base_step for k, v in tss.iteritems()}
nx.set_edge_attributes(g, new_tss, "t")

print("Vertices: %d" % g.number_of_nodes())
print("Edges: %d" % g.number_of_edges())

with open(output_json, "w") as wf:
  data = json_graph.node_link_data(g)
  json.dump(data, wf, indent=2)


