



class Grouping:
  
  def __init__(self):
    self.keyelem = []  ## Key element and prop value

  def parse(self, exprs):
    for expr in exprs:
      self.keyelem.append(Grouping._parse(expr))

  @staticmethod
  def _parse(expr):
    symbols = expr.split(".")
    if len(symbols) == 2:
      elemKey = symbols[0]
      propKey = symbols[1]
    else:
      elemKey = symbols
      propKey = None
    return (elemKey, propKey)

  
  def groupBy(self, results):
    keysize = len(self.keyelem)
    groupkeys = set()
    groups = {}  ## Group key -> result list
    
    for result in results:
      key = []
      for ke in self.keyelem:
        v = result.get_node_prop(ke[0], ke[1])
        key.append(v)
      if not key in groupkeys:
        groupkeys.add(key)
        groups[key] = []
      groups[key].append(result)
    
    return groups
    

