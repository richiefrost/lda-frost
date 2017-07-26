import Pyro4
import Pyro4.util
try:
    from Queue import Queue
except ImportError:
    from queue import Queue
from six import iteritems, itervalues
import numpy as np
import argparse
import sys
from worker import Worker
import utils
from time import time, sleep

LDA_WORKER_PREFIX = 'lda.worker'
LDA_SERVER_PREFIX = 'lda.server'
LDA_DISPATCHER_PREFIX = 'lda.dispatcher'

# timeout for the Queue object put/get blocking methods.
# it should theoretically be infinity, but then keyboard interrupts don't work.
# so this is really just a hack, see http://bugs.python.org/issue1360
HUGE_TIMEOUT = 365 * 24 * 60 * 60 # one year

class Dispatcher:
	def __init__(self):
		Pyro4.config.COMPRESSION = True

	@Pyro4.expose
	def initialize(self, V, K, alpha, beta, num_iter):
		self.K = K
		self.V = V

		# Get all the workers and servers on the network
		self.workers = {}
		with utils.getNS() as ns:
			self.callback = Pyro4.Proxy(ns.list(prefix=LDA_DISPATCHER_PREFIX)[LDA_DISPATCHER_PREFIX])
			self.server = Pyro4.Proxy(ns.list(prefix=LDA_SERVER_PREFIX)[LDA_SERVER_PREFIX])
			self.server.initialize(self.V, self.K)
			
			for name, uri in iteritems(ns.list(prefix=LDA_WORKER_PREFIX)):
				try:
					worker = Pyro4.Proxy(uri)
					workerId = len(self.workers)
					worker.initialize(self.callback, self.server, workerId, K, alpha, beta, num_iter)
					self.workers[workerId] = worker
				except Pyro4.errors.PyroError:
					print "Unresponsive worker at %s, deleting it from name server" % uri
					ns.remove(name)

		if not self.workers:
			raise RuntimeError('no workers found; run some lda_worker scripts on your machines first!')

		# Keep track of the number of words being processed on each worker
		self.load = np.zeros(len(self.workers), dtype=np.int64)
		print "Using %i workers" % len(self.workers)

	@Pyro4.expose
	def add_doc(self, dw_row):
		num_words = np.sum(dw_row)

		# Load balancing: Send this doc to the worker with the least number of words to compute right now
		which_worker = np.argmin(self.load)
		self.workers[which_worker].receive_dw_row(dw_row)
		
		# Update the load balancer to know this worker now has a heaver load
		self.load[which_worker] += num_words

	def join(self):
		while not all([worker.get_done() for worker in self.workers.values()]):
			sleep(0.5)

	@Pyro4.expose
	def train(self):
		for worker in self.workers.values():
			worker.start()
		print "All workers started, waiting for training to finish"
		self.join()
		print "Training finished"

		dt = []
		for worker in self.workers.values():
			dt += worker.get_dt()
		wt = self.server.get_wt()
		zt = self.server.get_zt()

		return wt, dt, zt
		
	
	@Pyro4.expose
	def num_workers(self):
		return len(self.workers)

	@Pyro4.expose
	@Pyro4.oneway
	def exit(self):
		import os
		os._exit(0)

	def finish(self):
		print time()
		for worker_id, worker in iteritems(self.workers):
			print "Terminating worker %i" % worker_id
			worker.exit()
		print "Terminating dispatcher"
		self.exit()


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--host", help="Nameserver hostname (default: %(default)s)", default=None)
	args = parser.parse_args()
	ns_conf = {"host": args.host}
	sys.excepthook = Pyro4.util.excepthook
	utils.pyro_daemon(LDA_DISPATCHER_PREFIX, Dispatcher(), ns_conf=ns_conf)

if __name__ == "__main__":
	main()
