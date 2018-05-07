import sys
import os.path
import collections
import matplotlib.pyplot as plt
import json
from networkx.readwrite import json_graph

argv = sys.argv
if len(argv) < 2:
	print("Usage: pythonw %s [JSON Graph]" % argv[0])
	exit(1)

input_json = argv[1]
base, ext = os.path.splitext(input_json)
output_png = base + ".png"

with open(input_json, "r") as f:
	data = json.load(f)

G = json_graph.node_link_graph(data)
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())
plt.xscale("log")
plt.yscale("log")
plt.title("Degree distribution: " + base)
plt.xlabel("degree")
plt.ylabel("count")
plt.plot(deg, cnt, "o")
plt.savefig(output_png)

