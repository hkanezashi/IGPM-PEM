
class Ordering:
  
  def __init__(self, expr):
    self.keyelem = Ordering.parse(expr)  ## Key element and prop value
  
  #def parse(self, expr):
  #  self.keyelem = Ordering.parse(expr)
  
  @staticmethod
  def parse(expr):
    symbols = expr.split(".")
    if len(symbols) == 2:
      elemKey = symbols[0]
      propKey = symbols[1]
    else:
      elemKey = symbols
      propKey = None
    return (elemKey, propKey)
  
  
  def orderBy(self, results, desc=False):
    elemKey = self.keyelem[0]
    propKey = self.keyelem[1]
    ordered = sorted(results, key=lambda r: r.get_node_prop(elemKey, propKey), reverse=desc)
    return ordered

