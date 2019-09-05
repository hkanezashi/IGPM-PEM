# G-Ray for multiple nodes
import Queue
import time
from multiprocessing.dummy import Pool
from multiprocessing import Manager, Value, Process


import pathos.pools as pp
import networkx as nx
from multiprocessing.managers import SyncManager


PORTNUM = 5000
AUTHKEY = "key"


def make_nums(N):
  nums = [999999999999]
  for i in xrange(N):
    nums.append(nums[-1] + 2)
  return nums


def make_server_manager(port, authkey):
  """ Create a manager for the server, listening on the given port.
      Return a manager object with get_job_q and get_result_q methods.
  """
  job_q = Queue.Queue()
  result_q = Queue.Queue()

  class JobQueueManager(SyncManager):
    pass
  
  JobQueueManager.register('get_job_q', callable=lambda: job_q)
  JobQueueManager.register('get_result_q', callable=lambda: result_q)
  
  manager = JobQueueManager(address=('', port), authkey=authkey)
  manager.start()
  print('Server started at port %s' % port)
  return manager


def runserver():
  # Start a shared manager server and access its queues
  manager = make_server_manager(PORTNUM, AUTHKEY)
  shared_job_q = manager.get_job_q()
  shared_result_q = manager.get_result_q()

  N = 999
  nums = make_nums(N)

  # The numbers are split into chunks. Each chunk is pushed into the job queue.
  chunksize = 20
  for i in range(0, len(nums), chunksize):
    shared_job_q.put(nums[i:i + chunksize])

  # Wait until all results are ready in shared_result_q
  numresults = 0
  resultdict = {}
  while numresults < N:
    outdict = shared_result_q.get()
    resultdict.update(outdict)
    numresults += len(outdict)

  # Sleep a bit before shutting down the server - to give clients time to
  # realize the job queue is empty and exit in an orderly way.
  time.sleep(2)
  manager.shutdown()


def factorizer_worker(job_q, result_q):
  """ A worker function to be launched in a separate process. Takes jobs from
      job_q - each job a list of numbers to factorize. When the job is done,
      the result (dict mapping number -> list of factors) is placed into
      result_q. Runs until job_q is empty.
  """
  while True:
    try:
      job = job_q.get_nowait()
      outdict = {n: factorize_naive(n) for n in job}
      result_q.put(outdict)
    except Queue.Empty:
      return


def mp_factorizer(shared_job_q, shared_result_q, nprocs):
  """ Split the work with jobs in shared_job_q and results in
      shared_result_q into several processes. Launch each process with
      factorizer_worker as the worker function, and wait until all are
      finished.
  """
  procs = []
  for i in range(nprocs):
    p = Process(target=factorizer_worker, args=(shared_job_q, shared_result_q))
    procs.append(p)
    p.start()

  for p in procs:
    p.join()

def make_client_manager(ip, port, authkey):
  class ServerQueueManager(SyncManager):
    pass

  ServerQueueManager.register('get_job_q')
  ServerQueueManager.register('get_result_q')
  manager = ServerQueueManager(address=(ip, port), authkey=authkey)
  manager.connect()
  print('Client connected to %s:%s' % (ip, port))
  return manager


def run_client():
  manager = make_client_manager("localhost", PORTNUM, AUTHKEY)
  job_q = manager.get_job_q()
  result_q = manager.get_result_q()
  mp_factorizer(job_q, result_q, 1)


def f(pid):
  count[pid] = nx.Graph()


manager = Manager()
count = manager.dict()

num_proc = 4

pool = Pool(4)
pool.map_async(f, list(range(num_proc)))

pool.close()
pool.join()

print(count)

