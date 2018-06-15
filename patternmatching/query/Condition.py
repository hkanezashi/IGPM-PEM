
LABEL = 'label'
TYPE = '_type'
PATH = '_path'

import logging

class Condition:

  @staticmethod
  def get_node_label(g, i):
    return g.nodes(data=True)[i].get(LABEL, '')
  
  @staticmethod
  def get_edges(g, src, dst):
    return g.edges[src, dst]
  
  """
  Get a dict between edge ID and label from specified src and dst
  """
  @staticmethod
  def get_edge_labels(g, src, dst):
    edges = g.get_edge_data(src, dst)
    labels = {}
    # print edges
    for eid, v in edges.iteritems():
      l = v[LABEL]
      labels[eid] = l
    return labels
  
  """
  Get a pair of edge ID and label from specified src and dst
  """
  @staticmethod
  def get_edge_label(g, src, dst):
    edges = g.edges(src, dst, keys=True)
    # print g.edge
    # print src, dst, edges
    for s, d, k, v in edges:
      if v is not None and LABEL in v:
        return k, v[LABEL]
    return None
  
  """
  Get all edge IDs from specified src, dst and label
  """
  @staticmethod
  def get_edges_from_label(g, src, dst, label):
    edges = g.edges(src, dst, keys=True)
    eids = []
    for s, d, k, v in edges:
      if v is not None and v[LABEL] == label:
        eids.append(k)
    return eids
  
  """
  Remove all edges with specified src, dst and label
  """
  @staticmethod
  def remove_edges_from_label(g, src, dst, label):
    eids = Condition.get_edges_from_label(g, src, dst, label)
    for eid in eids:
      del g.edges[src, dst, eid]
  
  """
  Remove an edge with specified src, dst and label
  """
  @staticmethod
  def remove_edge_from_label(g, src, dst, label):
    labels = Condition.get_edge_labels(g, src, dst)
    eid = None
    for k, v in labels.iteritems():
      if v == label:
        eid = k
        break
    if eid is not None:
      del g.edges[src, dst, eid]
    else:
      logging.warning("No such edges with specified label found: " + label)
  
  """
  Remove an edge with specified src, dst and ID
  If ID is None, remove an edge randomly
  """
  @staticmethod
  def remove_edge_from_id(g, src, dst, eid):
    if eid is None:
      for s, d, id, v in g.edges(src, dst, keys=True):
        eid = id  # Pick up the first ID
        break
    g.remove_edge(src, dst, eid)
  
  
  @staticmethod
  def has_edge_label(g, src, dst, label):
    if label is None or label == '':  ## Any label OK
      return True
    labels = Condition.get_edge_labels(g, src, dst)
    return label in labels.values()
    
  
  @staticmethod
  def get_node_prop(g, id, key):
    # print g.nodes(data=True)[id]
    return g.nodes(data=True)[id].get(key, '')

  @staticmethod
  def get_node_props(g, id):
    return g.nodes(data=True)[id]

  @staticmethod
  def get_edge_prop(g, src, dst, key):
    return g.edges[src, dst].get(key, '')
  
  @staticmethod
  def is_path(g, src, dst, id=0):
    return g.edges[src, dst, id].get(TYPE, '') == PATH
  
  @staticmethod
  def set_path(g, src, dst, id=0):
    g.edges[src, dst, id][TYPE] = PATH

  ## Check whether this vertex has same label and props
  @staticmethod
  def satisfies_node(g, i, label, prop):
    if label and Condition.get_node_label(g, i) != label:
      return False
    for k, v in prop.iteritems():
      if k != LABEL and Condition.get_node_prop(g, i, k) != v:
        return False
    return True
  
  ## Filter nodes in input graph by a label and properties (set them empty if no restrictions)
  @staticmethod
  def filter_nodes(g, label, prop, candidates=None):
    if candidates is None:
      nodes = g.nodes()
    else:
      nodes = candidates
    
    if label:
      nodes = [i for i in nodes if Condition.get_node_label(g, i) == label]
    for k, v in prop.iteritems():
      nodes = [i for i in nodes if Condition.get_node_prop(g, i, k) == v]
    return nodes
  
  """
  ## Filter edges in input graph by a label and properties (set them empty if no restrictions)
  @staticmethod
  def filter_edges(g, label, prop):
    edges = g.edges()
    if label:
      edges = [e for e in edges if g.get_edge_data(*e)[LABEL] == label]
    for k, v in prop.iteritems():
      edges = [e for e in edges if g.get_edge_data(*e)[k] == v]
    return edges
  """
  
  
  
