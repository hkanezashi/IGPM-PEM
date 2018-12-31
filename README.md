# IGPM-PEM
Scalable and Approximate Pattern Matching for Billion-Scale Property Graphs


## Load edgelist file
```bash
python load_edgelist.py [Edges/Step] [Steps] [InputEdgelist] [OutputJSON]
```
- Edges/Step: Number of edges to be added per step
- Steps: Number of total steps
- InputEdgelist: Edge list file (first and second columns indicate source and destination vertex ID)
- OutputJSON: Graph file in JSON format


## Configuration File (ini format)
- GraphJSON: Input Graph file in JSON format
- Steps: Number of total steps
- QueryArgs...: Options for the query graph
    - `--vertex` Vertex ID list. Example:`--vertex a b c`
    - `--edge` Edge (edgeID:sourceID:destinationID) list. Example:`--edge x:a:b y:b:c z:c:a`
    - `--vertexlabel` Vertex label restriction (vertexID:label). Example:`--vertexlabel a:cyan b:cyan c:cyan`
    - `--edgelabel` Edge label restriction (edgeID:label). Example: `--edgelabel x:yes y:yes z:yes`


## Batch
```bash
python patternmatching/gray/query_call.py [ConfFile]
```

## Incremental
```bash
export PYTHONPATH=$(pwd)
python patternmatching/gray/incremental/query_call.py [ConfFile]
```

## Incremental with PEM
```bash
export PYTHONPATH=$(pwd)
python patternmatching/gray/incremental/partial_execution_manager.py [ConfFile]
```


# Reference
Adaptive Pattern Matching with Reinforcement Learning for Dynamic Graphs. https://arxiv.org/abs/1812.10321
