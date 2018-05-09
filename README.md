# ScalablePM
Scalable and Approximate Pattern Matching for Billion-Scale Property Graphs


## Load edgelist file
```bash
python load_edgelist.py [Edges/Step] [Steps] [InputEdgelist] [OutputJSON]
```
- Edges/Step: Number of edges to be added per step
- Steps: Number of total steps
- InputEdgelist: Edge list file (first and second columns indicate source and destination vertex ID)
- OutputJSON: Graph file in JSON format

## Batch
```bash
export PYTHONPATH=$(pwd)
python patternmatching/gray/query_call.py [GraphJSON] [Steps] [QueryArgs...]
```
- GraphJSON: Input Graph file in JSON format
- Steps: Number of total steps
- QueryArgs...: Options for the query graph
    - `--vertex` Vertex ID list. Example:`--vertex a b c`
    - `--edge` Edge (edgeID:sourceID:destinationID) list. Example:`--edge x:a:b y:b:c z:c:a`
    - `--vertexlabel` Vertex label restriction (vertexID:label). Example:`--vertexlabel a:cyan b:cyan c:cyan`
    - `--edgelabel` Edge label restriction (edgeID:label). Example: `--edgelabel x:yes y:yes z:yes`


## Incremental
```bash
export PYTHONPATH=$(pwd)
python patternmatching/gray/incremental/query_call.py [GraphJSON] [Steps] [QueryArgs...]
```

## Incremental with PEM
```bash
export PYTHONPATH=$(pwd)
python patternmatching/gray/incremental/partial_execution_manager.py [GraphJSON] [Steps] [QueryArgs...]
```


