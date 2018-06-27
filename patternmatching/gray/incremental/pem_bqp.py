"""
Patrial Execution Manager
- Renforcement learning component to compute parameters for clustering
- Graph clustering by Louvain method
"""
import sys
import time
import logging
from ConfigParser import ConfigParser  # Use ConfigParser instead of configparser
import networkx as nx
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.policy import EpsGreedyQPolicy, GreedyQPolicy, BoltzmannQPolicy
from rl.agents.dqn import DQNAgent
from rl.memory import EpisodeParameterMemory, SequentialMemory

sys.path.append(".")

from patternmatching.gray.incremental.query_call import load_graph, parse_args
from patternmatching.gray.incremental.rl_model import GraphEnv


logging.basicConfig(level=logging.INFO)

argv = sys.argv
if len(argv) < 2:
  print("Usage: python %s [ConfFile]" % argv[0])
  exit(1)

conf = ConfigParser()
conf.read(argv[1])
graph_json = conf.get("G-Ray", "input_json")
base_step = int(conf.get("G-Ray", "base_steps"))
max_step = int(conf.get("G-Ray", "steps"))
args = conf.get("G-Ray", "query").split(" ")
time_limit = float(conf.get("G-Ray", "time_limit"))

graph = nx.Graph(load_graph(graph_json))
train_step = max_step / 2
test_step = max_step - train_step
# train_step = max_step
# test_step = max_step
query, cond, directed, groupby, orderby, aggregates = parse_args(args)


window_length = 5  # Should be up to 20 (too large length will not converge Q-values)
env = GraphEnv(graph, query, cond, base_step, train_step, time_limit, window_length)
nb_actions = env.action_space.n # len(env.action_space)
input_shape = env.observation_space.shape
print "Input shape:", input_shape

model = Sequential()
model.add(Flatten(input_shape=input_shape))
model.add(Dense(4))
model.add(Activation('relu'))
model.add(Dense(4))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())
print(env.observation_space)

# memory = EpisodeParameterMemory(limit=20, window_length=window_length)  # Non-episodic
memory = SequentialMemory(limit=20, window_length=window_length)

policy = BoltzmannQPolicy()  # Unstable

agent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=train_step,  # A3C TRPO
               target_model_update=1e-2, policy=policy)
agent.compile(Adam(lr=1e-2), metrics=['mae'])

st = time.time()
agent.fit(env, train_step)
ed = time.time()
print("Training: %f [s]" % (ed - st))

# Reset environment
# env.rewind()
# env = GraphEnv(graph, query, cond, base_step, test_step, time_limit, window_length)

st = time.time()
agent.test(env, nb_episodes=1)
ed = time.time()
print("Testing: %f [s]" % (ed - st))

