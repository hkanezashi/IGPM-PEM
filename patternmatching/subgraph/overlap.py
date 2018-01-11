"""
Create overlapping subgraphs from divided subgraphs
"""

import networkx as nx


def create_overlap(g, sg, depth):
  """
  Create an overlapping subgraph from a divided subgraph and an original input graph
  :param g: Original input graph
  :param sg: Divided subgraph
  :param depth: Depth of overlapping
  :return: Subgraph with overlapping
  
  :type g: nx.Graph
  :type sg: nx.Graph
  :type depth: int
  :rtype: nx.Graph
  """
  
  nodes = set(sg.nodes())
  visited = set()
  
  ## Find external vertices and edges with DFS
  ## https://networkx.github.io/documentation/stable/_modules/networkx/algorithms/traversal/depth_first_search.html#dfs_edges
  for start in sg.nodes():
    if start in visited:
      continue
    visited.add(start)
    stack = [(start, depth, iter(g[start]))]
    while stack:
      parent, depth_now, children = stack[-1]
      try:
        child = next(children)
        if child not in visited and child not in nodes:  # Already a member in the original subgraph?
          nodes.add(child)
          visited.add(child)  ## Add the next vertex
          if depth_now > 1:
            stack.append((child, depth_now - 1, iter(g[child])))
      except StopIteration:
        stack.pop()
  
  subgraph = g.subgraph(nodes)
  
  return subgraph
  
  