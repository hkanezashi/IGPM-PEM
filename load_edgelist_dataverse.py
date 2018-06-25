import csv
from datetime import datetime
import json
import networkx as nx
from networkx.readwrite import json_graph
import sys


epoch = datetime.utcfromtimestamp(0)
def convert_timestamp(ts_str):
  ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
  return int((ts - epoch).total_seconds()) / (24*60*60)  # sec --> day

argv = sys.argv
if len(argv) < 3:
  print("Usage: python %s [InputCSV] [OutputJSON]")
  exit(1)

input_csv = argv[1]
output_json = argv[2]

g = nx.MultiGraph()

with open(input_csv, "r") as rf:
  reader = csv.reader(rf, quotechar="'", delimiter=",")
  for row in reader:
    ts = row[0].replace("\"", "")
    src = int(row[1])
    dst = int(row[2].replace("\"", ""))
    sec = convert_timestamp(ts)
    
    g.add_edge(src, dst, add=sec)

tss = nx.get_edge_attributes(g, "add")
base_step = min(tss.values())
print("Base step: %d" % base_step)
new_tss = {k: v - base_step for k, v in tss.iteritems()}
nx.set_edge_attributes(g, new_tss, "add")

print("Vertices: %d" % g.number_of_nodes())
print("Edges: %d" % g.number_of_edges())

with open(output_json, "w") as wf:
  data = json_graph.node_link_data(g)
  json.dump(data, wf, indent=2)


