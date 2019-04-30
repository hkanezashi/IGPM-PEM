# IGPM-PEM

IGPM-PEM is the Scalable and Approximate Pattern Matching for Billion-Scale Property Graphs

For more details, please visit our [Wiki page](https://github.com/hkanezashi/IGPM-PEM/wiki).


# Launch commands and scripts

## Batch
```bash
python patternmatching/gray/query_call.py [ConfFile]
```

## Incremental (Naive version)
```bash
export PYTHONPATH=$(pwd)
python patternmatching/gray/incremental/query_call.py [ConfFile]
```

## Incremental (Adaptive version with PEM)
```bash
export PYTHONPATH=$(pwd)
python patternmatching/gray/incremental/partial_execution_manager.py [ConfFile]
```


# Reference
Hiroki Kanezashi, Toyotaro Suzumura, Dario Garcia-Gasulla, Min-Hwan and Satoshi Matsuoka, "Adaptive Pattern Matching with Reinforcement Learning for Dynamic Graphs", 25TH IEEE International Conference on High Performance Computing, Data, and Analytics (HiPC 2018) https://arxiv.org/abs/1812.10321

