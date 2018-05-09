# ScalablePM
Scalable and Approximate Pattern Matching for Billion-Scale Property Graphs


## Load edgelist file
```bash
python load_edgelist.py [Edges/Step] [Steps] [InputEdgelist] [OutputJSON]
```


## Batch
```bash
export PYTHONPATH=$(pwd)
python patternmatching/gray/query_call.py [GraphJSON] [Steps] [QueryArgs...]
```


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


