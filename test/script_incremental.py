from patternmatching.gray.incremental import query_call
import logging
logging.basicConfig(level=logging.INFO)

fname = "./sample/dynamic/test0.json"
qargs = "--vertex a b c --edge x:a:b y:b:c z:c:a --vertexlabel a:cyan b:cyan c:cyan".split(" ")
query_call.run_query(fname, qargs, False, False, 2)

fname = "./sample/dynamic/test1.json"
qargs = "--vertex a b c --edge x:a:b y:b:c z:c:a --vertexlabel a:cyan b:cyan c:cyan".split(" ")
query_call.run_query(fname, qargs, False, False, 3)

fname = "./sample/dynamic/test2.json"
qargs = "--vertex a b c --edge x:a:b y:b:c z:c:a --vertexlabel a:cyan b:cyan c:magenta".split(" ")
query_call.run_query(fname, qargs, False, False, 3)


fname = "./data/Congress1.json"
qargs = "--vertex a b c --edge x:a:b y:b:c z:c:a --vertexlabel a:cyan b:cyan c:cyan".split(" ")
query_call.run_query(fname, qargs, False, False, 10)
