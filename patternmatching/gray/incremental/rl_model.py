import networkx as nx
import community
import gym
from gym.spaces import Box, Discrete
import numpy as np
import statistics
import gray_incremental
import sys

from patternmatching.gray.incremental.query_call import get_seeds
from patternmatching.gray.parallel.gray_mp_inc import *

gym.envs.register(id='graphenv-v0', entry_point='GraphEnv')


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
      return [rev_part.values()[0]]
    
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
  

def get_recompute_nodes(graph, affected, min_size):
  part, rev_part = recursive_louvain(graph, min_size)
  nodes = set()
  
  affected_com = set()
  
  for n in affected:
    if not n in part:
      continue
    gid = part[n]
    mem = set(rev_part[gid])
    nodes.update(mem)
    affected_com.add(gid)
    
  return nodes, len(affected_com), len(rev_part)


def run_query_part(args):
  grm, edges, pid = args
  add_nodes = set([src for (src, dst) in edges] + [dst for (src, dst) in edges])
  affected_nodes = get_seeds(grm.graph, add_nodes, grm.community_size)
  grm.run_incremental_gray(edges, affected_nodes)
  num_patterns = len(grm.get_results())
  return num_patterns



class GraphEnv(gym.Env):
  """Reinforcement learning environment for graph object
  """
  
  def __init__(self, graph, query, cond, base_step, max_step, time_limit, window_length):
    """Constructor of an environment object for graph
    
    :param graph: Input data graph
    :param query: Query data graph
    :param cond: Condition object
    :param max_step: Number of iterations
    :param time_limit: Time limit of G-Ray iterations
    """
    super(gym.Env, self).__init__()
    
    ## Extract edge timestamp
    add_edge_timestamps = nx.get_edge_attributes(graph, "add")  # edge, time
    def dictinvert(d):
      inv = {}
      for k, v in d.iteritems():
        keys = inv.setdefault(v, [])
        keys.append(k)
      return inv
    self.add_timestamp_edges = dictinvert(add_edge_timestamps)  # time, edges

    self.step_list = sorted(list(self.add_timestamp_edges.keys()))
    start_step = self.step_list[0]

    ## Initialize base graph
    print("Initialize base graph")
    init_graph = nx.Graph()
    start_steps = self.step_list[0:base_step]
    init_edges = set()
    for start_step in start_steps:
      init_edges.update(set(self.add_timestamp_edges[start_step]))
      # init_graph.add_nodes_from(graph.nodes(data=True))
    init_graph.add_edges_from(init_edges)
    nx.set_edge_attributes(init_graph, 0, "add")
    
    print("Setup environment")
    self.action_space = Discrete(3)
    self.observation_space = Box(low=0, high=np.inf, shape=(window_length, 2), dtype=np.int32)  # Number of nodes, edges
    self.max_reward = 100.0
    self.reward_range = [-1., self.max_reward]
    self.grm = gray_incremental.GRayIncremental(graph, init_graph, query, graph.is_directed(), cond, time_limit)
    self.grm.run_gray()  # Initialization
    self.base_step = base_step
    self.max_step = max_step
    self.count = 0
    self.max_threshold, self.node_threshold = GraphEnv.compute_node_threshold(init_graph, query)
    self.reset()
  
  @staticmethod
  def compute_node_threshold(g, query):
    sizes = [len(wcc) for wcc in list(nx.weakly_connected_components(g.to_directed())) if len(wcc) > 1]
    max_size = max(sizes)
    init_size = query.number_of_nodes()
    print("Max: %d, Init: %d" % (max_size, init_size))
    return max_size, init_size
    
  def rewind(self):
    self.count = 0
  
  # def step(self, action):
  #
  #   if action == 0 and self.node_threshold > 1:
  #     self.node_threshold -= 1
  #   elif action == 1 and self.node_threshold < self.max_threshold:
  #     self.node_threshold += 1
  #   print("Community size: %d" % self.node_threshold)
  #
  #   t = self.count + self.base_step
  #   step = self.step_list[t]
  #   print("Step %d, index %d/%d" % (step, t, len(self.step_list)))
  #   add_edges = self.add_timestamp_edges[step]
  #   add_nodes = set([src for (src, dst) in add_edges] + [dst for (src, dst) in add_edges])
  #   affected_nodes, affected_com, total_com = get_recompute_nodes(self.grm.graph, add_nodes, self.node_threshold)
  #   self.grm.run_incremental_gray(add_edges, affected_nodes)
  #   self.count += 1
  #   stop = (self.count >= self.max_step)
  #
  #   def get_observation():
  #     nodes = self.grm.graph.number_of_nodes()
  #     edges = self.grm.graph.number_of_edges()
  #
  #     total_density = float(edges) / nodes
  #     com_density = float(affected_com) / total_com
  #     return np.array([total_density, com_density])
  #
  #   sys.stdout.flush()
  #   return get_observation(), self.grm.get_reward(self.max_reward), stop, {}

  def step(self, action):
    num_proc = 1
    
    if action == 0 and self.node_threshold > 1:
      self.node_threshold -= 1
    elif action == 1 and self.node_threshold < self.max_threshold:
      self.node_threshold += 1
    print("Community size: %d" % self.node_threshold)
    
    t = self.count + self.base_step
    step = self.step_list[t]
    print("Step %d, index %d/%d" % (step, t, len(self.step_list)))
    add_edges = self.add_timestamp_edges[step]
    add_nodes = set([src for (src, dst) in add_edges] + [dst for (src, dst) in add_edges])

    st = time.time()
    procs = list()
    edge_chunks = split_list(add_edges, num_proc)
    for pid in range(num_proc):
      print(len(edge_chunks[pid]))
      procs.append(Process(target=run_query_part, args=((self.grm, edge_chunks[pid], pid),)))
    for proc in procs:
      proc.start()
    for proc in procs:
      proc.join()
    ed = time.time()
    elapsed = ed - st
    print("Time at step %d: %f[s]" % (t, elapsed))
    
    affected_nodes, affected_com, total_com = get_recompute_nodes(self.grm.graph, add_nodes, self.node_threshold)
    self.count += 1
    stop = (self.count >= self.max_step)
    
    def get_observation():
      nodes = self.grm.graph.number_of_nodes()
      edges = self.grm.graph.number_of_edges()
      
      total_density = float(edges) / nodes
      com_density = float(affected_com) / total_com
      return np.array([total_density, com_density])
    
    sys.stdout.flush()
    return get_observation(), self.grm.get_reward(self.max_reward), stop, {}
    
  
  def reset(self):
    return self.grm.get_observation()
  
  
  def render(self, mode='human', close=False):
    # print self.count, self.observation_space, self.node_threshold
    pass
  
  def close(self):
    pass
  
  def seed(self, seed=None):
    pass
  


