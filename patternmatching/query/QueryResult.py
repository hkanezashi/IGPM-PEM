

from Condition import *

class QueryResult:
  
  def __init__(self, graph, nodemap):
    self.graph = graph  ## Extracted graph from input
    self.nodemap = nodemap  ## Node symbol mapping (query -> input)
  
  def get_node_id(self, id_query):
    return self.nodemap[id_query]
  
  def get_node_label(self, id_query):
    id_result = self.get_node_id(id_query)
    return Condition.get_node_label(self.graph, id_result)
  
  def get_node_prop(self, id_query, key):
    id_result = self.get_node_id(id_query)
    value = Condition.get_node_prop(self.graph, id_result, key)
    # print id_query, id_result, key, value
    return value
  
  def get_graph(self):
    return self.graph
  
  


