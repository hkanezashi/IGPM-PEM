"""
Parse graph query and get eval conditions

# References:
# http://stackoverflow.com/questions/11133339/parsing-a-complex-logical-expression-in-pyparsing-in-a-binary-tree-fashion
# http://stackoverflow.com/questions/33532451/pyparsing-python-binary-boolean-expression-to-xml-nesting-issue-2-7-10
# http://pyparsing.wikispaces.com/file/view/simpleArith.py/30268305/simpleArith.py
# http://qiita.com/knoguchi/items/ee949989d0a9f04bee6f
# http://qiita.com/knoguchi/items/6f9b7383b7252a9ebdad
"""


import logging
import pyparsing as pp
import networkx as nx

from patternmatching.query.Condition import *


LPAR,RPAR = map(pp.Suppress,"()")
numvalue = pp.Regex(r"\d+(\.\d*)?([eE][+-]?\d+)?")
term = pp.Forward()
factor = pp.Forward()

addsub = pp.oneOf('+ -')
muldiv = pp.oneOf('* /')
compare = pp.Regex(">=|<=|!=|>|<|==").setName("compare")
NOT_ = pp.Keyword("NOT").setName("NOT")
AND_ = pp.Keyword("AND").setName("AND")
OR_ = pp.Keyword("OR").setName("OR")

symbol = pp.Word(pp.alphas).setName("symbol")
propsymbol = pp.Group(symbol + "." + symbol).setName("propsymbol")
formula = pp.Optional(addsub) + term + pp.ZeroOrMore(addsub + term)
term << (factor + pp.ZeroOrMore(muldiv + factor))
factor << (numvalue | propsymbol | LPAR + formula + RPAR)

factor = numvalue | propsymbol
# condition = pp.Group(factor + compare + factor)

formula = pp.infixNotation(factor,[
  (muldiv, 2, pp.opAssoc.LEFT, ),
  (addsub, 2, pp.opAssoc.LEFT, ),
])

condition = pp.infixNotation(formula,[
  (compare, 2, pp.opAssoc.LEFT, ),
])

expr = pp.operatorPrecedence(condition,[
      ("NOT", 1, pp.opAssoc.RIGHT, ),
      ("AND", 2, pp.opAssoc.LEFT, ),
      ("OR", 2, pp.opAssoc.LEFT, ),
])

# print expr.parseString("x.a > 7 AND x.b < 8 OR x.c == 4")
# print expr.parseString("x.a > 7 AND x.b < 8 OR x.c * 2 + 1 == 4")

class ConditionParser:
  
  def __init__(self, eq_str):
    """
    :param eq_str: Query string
    :type eq_str: str
    """
    self.expr = expr.parseString(eq_str)[0]
    
  def eval(self, g, nodemap=None, expr=None):
    if nodemap is None:
      nodemap = dict()
    if expr is None:
      expr = self.expr
    logging.info(expr)
    
    if len(expr) == 1:
      elem = expr[0]
      if elem.isdigit():
        return float(elem)
      else:
        return elem
    elif len(expr) == 2:
      if expr[0] == "NOT":
        return not self.eval(g, nodemap, expr[1])
      else:
        logging.warning("Unknown operator accepts single value: " + expr[0])
        return None
    elif len(expr) == 3:
      op = expr[1]
      LEFT = self.eval(g, nodemap, expr[0])
      RIGHT = self.eval(g, nodemap, expr[2])
      if op == "AND":
        return LEFT and RIGHT
      elif op == "OR":
        return LEFT or RIGHT
      elif op == ">=":
        return LEFT >= RIGHT
      elif op == "<=":
        return LEFT <= RIGHT
      elif op == "!=":
        return LEFT != RIGHT
      elif op == ">":
        return LEFT > RIGHT
      elif op == "<":
        return LEFT < RIGHT
      elif op == "==":
        return LEFT == RIGHT
      elif op == "+":
        return LEFT + RIGHT
      elif op == "-":
        return LEFT - RIGHT
      elif op == "*":
        return LEFT * RIGHT
      elif op == "/":
        return LEFT / RIGHT
      elif op == ".":
        if LEFT in nodemap:  ## Replace symbols to output ones
          LEFT = nodemap[LEFT]
        if RIGHT == LABEL:
          value = Condition.get_node_label(g, LEFT)
          if value.isdigit():
            return float(value)
          return value
        else:
          value = Condition.get_node_prop(g, LEFT, RIGHT)
          if value.isdigit():
            return float(value)
          return value
      else:
        logging.warning("Unknown operator: " + op)
        return None
    else:
      logging.warning("Invalid syntax: too many elements")
      return None

if __name__ == "__main__":
  logging.basicConfig(level=logging.DEBUG)
  cp = ConditionParser("x.a > 7 AND x.b < 8 OR x.c * 2 - 6 == 4")
  g = nx.Graph()
  g.add_node("y", a="10", b="5", c="2")
  node_map = {"x": "y"} ## Node symbol (query -> output)
  ret = cp.eval(g, node_map)
  print(ret)


