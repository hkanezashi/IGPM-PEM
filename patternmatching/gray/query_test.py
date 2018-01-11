import logging

import query_call

### Label -> matplotlib color string
label_color = {'cyan': 'c', 'magenta': 'm', 'yellow': 'y', 'white': 'w'}

logging.basicConfig(level=logging.INFO)

test_cases = [
  ("sample/test1.gml",    # Case 1-1
   "--vertex a b c d e --edge x:a:b y:b:c z:c:d w:d:e --vertexlabel a:cyan b:cyan c:cyan d:cyan e:cyan",
   [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 2, 3, 4, 5], [0, 2, 3, 4, 5]]),
  ("sample/test1.gml",    # Case 1-2
   "--vertex a b c --edge x:a:b y:b:c z:c:a --vertexlabel a:cyan b:cyan c:cyan",
   [[0, 1, 3], [2, 4, 5]]),
  ("sample/test1.gml",    # Case 1-3
   "--vertex a b c d --edge x:a:b y:b:c z:c:d w:d:a --vertexlabel a:cyan b:cyan c:cyan d:cyan",
   [[0, 2, 3, 4]]),
  ("sample/test2.gml",    # Case 1-4
   "--vertex a b c --edge x:a:b y:b:c z:c:a --vertexlabel a:cyan b:cyan c:magenta",
   [[2, 4, 5]]),
  ("sample/test1p.gml",   # Case 1-5
   "--vertex a b c --edge x:a:b y:b:c z:c:a --aggregate MAX:a.score",
   [[0, 1, 3], [0, 3, 4], [2, 3, 4], [2, 4, 5]]),
  ("sample/test4.gml",    # Case 1-6
   "--vertex a b --edge x:a:b y:a:b --edgelabel x:yes y:no",
   [[0, 1], [2, 4]]),
  ("sample/test1d.gml",   # Case 2-1
   "--vertex a b c --edge x:a:b y:a:c z:b:c --directed --vertexlabel a:cyan b:cyan c:cyan",
   [[0, 1, 3], [2, 4, 5]]),
  ("sample/test2d.gml",   # Case 2-2
   "--vertex a b c --edge x:a:b y:a:c z:b:c --directed --vertexlabel a:cyan b:cyan c:magenta",
   [[2, 4, 5]]),
  ("sample/test1p.gml",   # Case 3-1
   "--vertex a b c --edge x:a:b y:a:c z:b:c --vertexlabel a:cyan b:cyan c:cyan --edgelabel x:yes y:yes z:yes",
   [[0, 3, 4]]),
  ("sample/line.gml",     # Case 4-1
   "--vertex a b --path x:a:b --vertexlabel a:cyan b:cyan --edgelabel x:yes",
   [[0, 1], [0, 1, 2], [0, 1, 2, 3], [1, 2], [1, 2, 3], [2, 3]]),
  ("sample/line.gml",     # Case 4-2
   "--vertex a b c --path x:a:b --edge y:b:c --vertexlabel a:cyan b:cyan c:cyan --edgelabel x:yes y:yes",
   [[0, 1, 2], [0, 1, 2, 3], [1, 2, 3]]),
  ("sample/test0p.gml",   # Case 4-3
   "--vertex a b c --edge x:a:b y:b:c --path z:a:c --vertexlabel a:cyan b:cyan c:cyan --edgelabel x:yes y:yes z:yes",
   [[0, 1, 2, 3]]),
  ("sample/test1p.gml",   # Case 4-4
   "--vertex a b c --edge x:a:b y:b:c --path z:c:a --vertexlabel a:cyan b:cyan c:cyan --edgelabel x:yes y:yes z:yes",
   [[0, 3, 4], [2, 3, 4, 5], [0, 2, 3, 4, 5]]),
]


for tc in test_cases:
  gfile = tc[0]
  qstr = tc[1]
  ans = tc[2]
  
  print "Input Graph File:", gfile
  print "Query Options:", qstr
  qargs = qstr.split(" ")
  logging.basicConfig(level=logging.WARNING)
  results = query_call.run_query(gfile, qargs)
  SUCCESS = True
  
  num_results = len(results)
  num_ans = len(ans)
  if num_results != num_ans:
    print "FAILED: the number of result graphs is different", num_results, " expected", num_ans
    SUCCESS = False
    continue
  for qresult in results:
    result = qresult.get_graph()
    # print result.nodes()
    
  print SUCCESS


