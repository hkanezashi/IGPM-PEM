

import logging

class Aggregator:
  
  def __init__(self, expr):
    self.op = Aggregator.parse(expr)
  
  #def parse(self, expr):
  #  self.op = self.parse(expr)

  @staticmethod
  def parse(expr):
    symbols = expr.split(":")
    opKey = symbols[0]
    if opKey == "COUNT":
      elemKey = None
      propKey = None
    else:
      elems = symbols[1].split(".")
      if len(elems) == 2:
        elemKey = elems[0]
        propKey = elems[1]
      else:
        elemKey = elems[0]
        propKey = None
    return opKey, elemKey, propKey

  def get_result(self, results):
    num_results = len(results)
    if num_results == 0:
      logging.warning("No result subgraphs")
      return None
    
    opKey = self.op[0]
    elemKey = self.op[1]
    propKey = self.op[2]
    
    if opKey == "COUNT":
      return num_results
    elif opKey == "MIN":
      return min(float(x.get_node_prop(elemKey, propKey)) for x in results.values())
    elif opKey == "MAX":
      return max(float(x.get_node_prop(elemKey, propKey)) for x in results.values())
    elif opKey == "SUM":
      return sum(float(x.get_node_prop(elemKey, propKey)) for x in results.values())
    elif opKey == "AVG":
      return sum(float(x.get_node_prop(elemKey, propKey)) for x in results.values()) / float(num_results)
    else:
      logging.warning("Unknown operator: " + opKey)
      return None

