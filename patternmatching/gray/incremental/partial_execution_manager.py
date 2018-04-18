"""
Patrial Execution Manager
- Renforcement learning component to compute parameters for clustering
- Graph clustering by Louvain method
"""
import sys
import networkx as nx
import community  # http://python-louvain.readthedocs.io/en/latest/

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from rl.agents.cem import CEMAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy

from patternmatching.gray.incremental.query_call import load_graph, parse_args
from patternmatching.gray.incremental.rl_model import GraphEnv


argv = sys.argv

if len(argv) < 4:
  print("Usage: python %s GraphJSON MaxStep QueryArgs...")

graph = load_graph(argv[1])
max_step = int(argv[2])
query, cond, directed, groupby, orderby, aggregates = parse_args(argv[3:])

env = GraphEnv(graph, query, cond, max_step)

nb_actions = len(env.action_space)

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
agent = CEMAgent(model, nb_actions, memory, nb_steps_warmup=5, target_model_update=1e-2, policy=policy)
agent.compile()

agent.fit(env, max_step)
agent.test(env, max_step)


class PEM:
  
  def __init__(self, g):
    """
    :type g: nx.Graph
    :param g: Input data graph
    """
    self.g = g  # Input data graph
    self.model = None  # Learning model
    self.agent = None  # Reinforcement learning agent (from keras-rl)
    
  
  def get_recompute_nodes(self, affected, resolution):
    """Get nodes for recomputations
    
    :param affected: Affected nodes for graph updates
    :param resolution: A parameter to change the size of the communities, represents the time described in
        "Laplacian Dynamics and Multiscale Modular Structure in Networks", R. Lambiotte, J.-C. Delvenne, M. Barahona
        https://arxiv.org/pdf/0812.1770.pdf
    :return: Set of nodes for recomputations
    """
    part = community.best_partition(self.g)  # node, partID
    
    nodes = set()
    
    return nodes
    
    


