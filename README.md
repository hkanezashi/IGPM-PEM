# IGPM-PEM
Scalable and Approximate Pattern Matching for Billion-Scale Property Graphs


## Load edgelist file
```bash
python load_edgelist.py [Edges/Step] [Steps] [InputEdgelist] [OutputJSON]
```
- Edges/Step: Number of edges to be added per step
- Steps: Number of total steps
- InputEdgelist: Edge list file (first and second columns indicate source and destination vertex ID)
- OutputJSON: Graph file in JSON format used for the graph pattern matching process


## Configuration File (ini format)
- GraphJSON: Input Graph file in JSON format
- Steps: Number of total steps
- QueryArgs...: Options for the query graph
    - `--vertex` Vertex ID list. Example:`--vertex a b c`
    - `--edge` Edge (edgeID:sourceID:destinationID) list. Example:`--edge x:a:b y:b:c z:c:a`
    - `--vertexlabel` Vertex label restriction (vertexID:label). Example:`--vertexlabel a:cyan b:cyan c:cyan`
    - `--edgelabel` Edge label restriction (edgeID:label). Example: `--edgelabel x:yes y:yes z:yes`



# Launch commands and scripts

## Batch
```bash
python patternmatching/gray/query_call.py [ConfFile]
```

## Incremental (Naive)
```bash
export PYTHONPATH=$(pwd)
python patternmatching/gray/incremental/query_call.py [ConfFile]
```

## Incremental (Adaptive with PEM)
```bash
export PYTHONPATH=$(pwd)
python patternmatching/gray/incremental/partial_execution_manager.py [ConfFile]
```

## Running IGPM-PEM in [TSUBAME3.0](https://www.t3.gsic.titech.ac.jp/en)
1. Assign a computation node: `qrsh -l f_node=1 -l h_rt=0:10:00`
1. Load modules: `module load python-extension`
1. Launch scripts as above


# Reference
Hiroki Kanezashi, Toyotaro Suzumura, Dario Garcia-Gasulla, Min-Hwan and Satoshi Matsuoka, "Adaptive Pattern Matching with Reinforcement Learning for Dynamic Graphs", 25TH IEEE International Conference on High Performance Computing, Data, and Analytics (HiPC 2018) https://arxiv.org/abs/1812.10321

