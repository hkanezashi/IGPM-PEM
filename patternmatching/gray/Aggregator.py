

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
    return (opKey, elemKey, propKey)

  def get_result(self, results):
    if not results:
      logging.warning("No result subgraphs")
      return None
    
    opKey = self.op[0]
    elemKey = self.op[1]
    propKey = self.op[2]
    
    if opKey == "COUNT":
      return len(results)
    elif opKey == "MIN":
      return min(float(x.get_node_prop(elemKey, propKey)) for x in results)
    elif opKey == "MAX":
      """
      print elemKey, propKey
      for x in results:
        print x.get_graph().nodes()
      """
      return max(float(x.get_node_prop(elemKey, propKey)) for x in results)
    elif opKey == "SUM":
      return sum(float(x.get_node_prop(elemKey, propKey)) for x in results)
    elif opKey == "AVG":
      return sum(float(x.get_node_prop(elemKey, propKey)) for x in results) / float(len(results))
    else:
      logging.warning("Unknown operator: " + opKey)
      return None

