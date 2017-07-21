import Pyro4
import Pyro4.util
try:
    from Queue import Queue
except ImportError:
    from queue import Queue
from six import iteritems, itervalues
import numpy as np
import sys
from worker import Worker
import utils
from time import time

LDA_WORKER_PREFIX = 'lda.worker'
LDA_DISPATCHER_PREFIX = 'lda.dispatcher'

# timeout for the Queue object put/get blocking methods.
# it should theoretically be infinity, but then keyboard interrupts don't work.
# so this is really just a hack, see http://bugs.python.org/issue1360
HUGE_TIMEOUT = 365 * 24 * 60 * 60 # one year

class Dispatcher:
	@Pyro4.expose
	def initialize(self, V, K, alpha, beta, num_iter, check_every=20):
		# Meant for coordinating updates
		self.tokens_received = 0
		self.check_every = check_every
		self.token_queue = Queue()
		self.K = K
		self.V = V
		self.iterations_left = np.repeat(num_iter, V + 1)
		# Get all the workers and servers on the network, but don't initialize them yet
		self.workers = {}
		with utils.getNS() as ns:
			self.callback = Pyro4.Proxy(ns.list(prefix=LDA_DISPATCHER_PREFIX)[LDA_DISPATCHER_PREFIX])
			print "self.callback", self.callback
			for name, uri in iteritems(ns.list(prefix=LDA_WORKER_PREFIX)):
				try:
					worker = Pyro4.Proxy(uri)
					workerId = len(self.workers)
					worker.initialize(self.callback, workerId, K, alpha, beta, num_iter)
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

	@Pyro4.expose
	def get_token(self):
		return self.token_queue.get()

	@Pyro4.expose
	def add_initial_token(self, token):
		self.token_queue.put(token)

	@Pyro4.expose
	def receive_token(self, token, worker_id):
		self.iterations_left[token[0]] -= 1
		if self.iterations_left[token[0]] > 0:
			self.token_queue.put(token)
		self.tokens_received += 1
		if self.tokens_received % self.check_every == 0:
			total = np.sum(self.iterations_left)
			#print "Total iterations left: %i" % total
			if total == 0:
				self.finish()
		self.workers[worker_id].request_token()

	@Pyro4.expose
	def train(self):
		print time()
		for worker in self.workers.values():
			worker.start()
			worker.request_token()
	
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
	sys.excepthook = Pyro4.util.excepthook
	with Pyro4.locateNS() as ns:
		with Pyro4.Daemon() as daemon:
			name = LDA_DISPATCHER_PREFIX
			uri = daemon.register(Dispatcher(), name)
			ns.remove(name)
			ns.register(name, uri)
			print "Dispatcher ready at %s" % uri
			daemon.requestLoop()

if __name__ == "__main__":
	main()

