import sys
from configparser import ConfigParser
import networkx as nx
from multiprocessing import Manager, Pool

sys.path.append(".")
sys.setrecursionlimit(1000)
from patternmatching.gray.parallel import gray_parallel

args = sys.argv
if len(args) < 2:
  print("Usage: python %s [ConfFile]" % args[0])
  sys.exit(1)

conf = ConfigParser()
conf.read(args[1])

gfile = conf.get("G-Ray", "input_json")
steps = int(conf.get("G-Ray", "steps"))
qargs = conf.get("G-Ray", "query").split(" ")
time_limit = float(conf.get("G-Ray", "time_limit"))
num_proc = int(conf.get("G-Ray", "num_proc"))
print("Graph file: %s" % gfile)
print("Query args: %s" % str(qargs))
print("Number of proc: %d" % num_proc)

gray_parallel.run_parallel_gray(gfile, qargs, num_proc)




# num = 10
# g = nx.complete_graph(num)
#
# manager = Manager()
# patterns = manager.dict()
#
# def get_egonet(n):
#   patterns[n] = nx.ego_graph(g, n).nodes()
#
# pool = Pool(2)
# pool.map(get_egonet, g.nodes())
# pool.close()
# pool.join()
#
# print patterns



