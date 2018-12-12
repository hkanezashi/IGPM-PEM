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

policies = {
  "bqp": BoltzmannQPolicy(),  # Unstable
  "gqp": GreedyQPolicy(),
  "egqp": EpsGreedyQPolicy(eps=0.1)  # eps should be around 0.1
}

window_length = 5  # Should be less than 20 (too large value will not converge Q-values)
memories = {
  "epm": EpisodeParameterMemory(limit=20, window_length=window_length),  # Non-episodic
  "sm": SequentialMemory(limit=20, window_length=window_length)  # should use this
}

argv = sys.argv
if len(argv) < 4:
  print("Usage: python %s [ConfFile] [Policy] [Memory]" % argv[0])
  exit(1)


policy_name = argv[2]
if not policy_name in policies:
  print("Please specify correct policy name: %s" % str(policies.keys()))
  exit(1)
policy = policies[policy_name]

memory_name = argv[3]
if not memories in memory_name:
  print("Please specify correct memory name: %s" % str(memories.keys()))
  exit(1)
memory = memories[memory_name]


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
query, cond, directed, groupby, orderby, aggregates = parse_args(args)



env = GraphEnv(graph, query, cond, base_step, train_step, time_limit, window_length)
nb_actions = env.action_space.n
input_shape = env.observation_space.shape
print "Input shape:", input_shape

# Build the neural network
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

agent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=train_step,
               target_model_update=1e-2, policy=policy)
agent.compile(Adam(lr=1e-2), metrics=['mae'])

st = time.time()
agent.fit(env, train_step)
ed = time.time()
print("Training: %f [s]" % (ed - st))

# env.reset()
env = GraphEnv(graph, query, cond, base_step, test_step, time_limit, window_length)

st = time.time()
agent.test(env, nb_episodes=1)
ed = time.time()
print("Testing: %f [s]" % (ed - st))

