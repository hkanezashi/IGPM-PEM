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

sys.path.append(".")

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
input_shape = env.observation_space.shape
print "Input shape:", input_shape

model = Sequential()
model.add(Flatten(input_shape=input_shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())
print(env.observation_space)

memory = SequentialMemory(limit=50, window_length=1)
agent = CEMAgent(model, nb_actions, memory, nb_steps_warmup=5)
agent.compile()

agent.fit(env, max_step)
agent.test(env, max_step)


