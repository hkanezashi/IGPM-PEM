# IGPM-PEM

IGPM-PEM is the scalable and approximate pattern matching for billion-scale property graphs.

**Note: Our [Wiki page](https://github.com/hkanezashi/IGPM-PEM/wiki) is still under construction, so please refer this README.md page first.**


# Software requirements
**Note: The Python scripts in this repository are implemented for Python 3.7 and NetworkX 2.3.**

- Python 3.7 with the following packages (see `requirements.txt`)
  - python-louvain
  - scikit-learn
  - numpy
  - scipy
  - pyparsing
  - matplotlib
  - networkx (2.3 or later)
  - metis
  - keras
  - keras-rl
  - gym
  - pathos
  - statistics
  - tensorflow


# Configuration file
A configuration file (INI format) is required to specify parameters such as input graph file (edge list) path, number of incremental process steps and the query pattern graph.
Here is an example of the configuration file content (`gray.ini`).

```ini
[Log]
profile = False
level = info

[G-Ray]
input_json = sample/test4.json
base_steps = 100
steps = 10
query = --vertex a b c --edge x:a:b y:b:c z:c:a
time_limit = 0.0
num_proc = 1
```

The configuration file has two sections: "Log" and "G-Ray".
- `Log`: Logging configurations
  - `profile`: Whether it enables profiling with cProfile
  - `level`: Logging level (debug, info, warn, etc.)
- G-Ray
  - `input_json`: Input graph file path as JSON format. See [NetworkX documentation](https://networkx.github.io/documentation/stable/reference/readwrite/json_graph.html) for details.
  - `base_steps`: Start step
  - `steps`: Number of steps for input graph updates
  - `query`: Options for the query graph
  - `time_limit`: Time limit (seconds) of pattern matching iterations for each step. You can disable the time limit feature with a non-positive value.
  - `num_proc`: (Experimental) number of processes for parallel executions

# Launch commands and scripts

## Batch
```bash
python3 patternmatching/gray/query_call.py [ConfFile]
```

## Incremental (Naive version)
```bash
python3 patternmatching/gray/incremental/query_call.py [ConfFile]
```

## Incremental (Adaptive version with PEM)
```bash
python3 patternmatching/gray/incremental/pem_egqp.py [ConfFile]
```


# Reference
Hiroki Kanezashi, Toyotaro Suzumura, Dario Garcia-Gasulla, Min-hwan Oh and Satoshi Matsuoka, "Adaptive Pattern Matching with Reinforcement Learning for Dynamic Graphs", 25TH IEEE International Conference on High Performance Computing, Data, and Analytics (HiPC 2018)
- arXiv: [1812.10321](https://arxiv.org/abs/1812.10321)
- DOI: [10.1109/HiPC.2018.00019](https://doi.org/10.1109/HiPC.2018.00019)

