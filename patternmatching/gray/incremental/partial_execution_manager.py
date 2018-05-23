"""
Patrial Execution Manager
- Renforcement learning component to compute parameters for clustering
- Graph clustering by Louvain method
"""
import sys
import time
import logging
from configparser import ConfigParser

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.policy import BoltzmannQPolicy
from rl.agents.dqn import DQNAgent
from rl.memory import EpisodeParameterMemory

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
max_step = int(conf.get("G-Ray", "steps"))
args = conf.get("G-Ray", "query").split(" ")
time_limit = float(conf.get("G-Ray", "time_limit"))

graph = load_graph(graph_json)
train_step = max_step / 2
test_step = max_step - train_step
query, cond, directed, groupby, orderby, aggregates = parse_args(args)

# init_graph = get_init_graph(graph)
env = GraphEnv(graph, query, cond, train_step, time_limit)
nb_actions = env.action_space.n # len(env.action_space)
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

memory = EpisodeParameterMemory(limit=50, window_length=1)
# agent = CEMAgent(model, nb_actions, memory, nb_steps_warmup=5)
# agent.compile()
policy = BoltzmannQPolicy()
agent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=train_step,
               target_model_update=1e-2, policy=policy)
agent.compile(Adam(lr=1e-3), metrics=['mae'])

st = time.time()
agent.fit(env, train_step)
ed = time.time()
print("Training: %f [s]" % (ed - st))

st = time.time()
agent.test(env, test_step)
ed = time.time()
print("Testing: %f [s]" % (ed - st))

