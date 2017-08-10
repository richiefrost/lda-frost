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
LDA_DISPATCHER_PREFIX = 'lda.dispatcher'

# timeout for the Queue object put/get blocking methods.
# it should theoretically be infinity, but then keyboard interrupts don't work.
# so this is really just a hack, see http://bugs.python.org/issue1360
HUGE_TIMEOUT = 365 * 24 * 60 * 60 # one year

class Dispatcher:
	@Pyro4.expose
	def initialize(self, V, K, alpha, beta, num_iter, check_every=50):
		# Meant for coordinating updates
		self.check_every = check_every
		self.K = K
		self.V = V
		self.finished_tokens = {}
		# Get all the workers and servers on the network, but don't initialize them yet
		self.workers = {}
		with utils.getNS() as ns:
			self.callback = Pyro4.Proxy(ns.list(prefix=LDA_DISPATCHER_PREFIX)[LDA_DISPATCHER_PREFIX])
			for name, uri in iteritems(ns.list(prefix=LDA_WORKER_PREFIX)):
				try:
					worker = Pyro4.Proxy(uri)
					workerId = len(self.workers)
					worker.initialize(self.callback, workerId, K=K, V=V, alpha=alpha, beta=beta, num_iter=num_iter)
					self.workers[workerId] = worker
				except Pyro4.errors.PyroError:
					print "Unresponsive worker at %s, deleting it from name server" % uri
					ns.remove(name)

		if not self.workers:
			raise RuntimeError('no workers found; run some lda.worker scripts on your machines first!')

		for worker in self.workers.values():
			worker.include_workers(self.workers)

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
	def create_corpus(self):
		for worker in self.workers.values():
			worker.create_corpus()
		self.token_mappings = {word: set() for word in range(self.V)}
		for worker in self.workers.values():
			for word in worker.get_uniques():
				self.token_mappings[word].add(worker.get_worker_id())
		#for worker in self.workers.values():
		#	worker.create_mappings()

	@Pyro4.expose
	@Pyro4.oneway
	def remove_worker_mapping(self, word, worker_id):
		self.token_mappings[word].discard(worker_id)

	@Pyro4.expose
	def get_token_mappings(self, word):
		return list(self.token_mappings[word])

	@Pyro4.expose
	def receive_finished_token(self, token):
		self.finished_tokens[token[0]] = token[1]

	@Pyro4.expose
	@Pyro4.oneway
	def add_initial_token(self, token):
		if token[0] == 0:
			which_worker = 0
		else:
			mappings = list(self.token_mappings[token[0] - 1])
			which_worker = mappings[0]
		self.workers[which_worker].receive_token(token)

	def join(self):
		while not all(worker.check_done() for worker in self.workers.values()):
			sleep(0.5)

	@Pyro4.expose
	def wait(self):
		self.join()

		print "Passed:"
		for worker in self.workers.values():
			print "Worker %i:" % worker.get_worker_id(), worker.tokens_passed()

		print
		print "Received"
		for worker in self.workers.values():
			print "Worker %i:" % worker.get_worker_id(), worker.tokens_received()

		print
		print "Dead Received"
		for worker in self.workers.values():
			print "Worker %i:" % worker.get_worker_id(), worker.get_dead_received()

		for worker in self.workers.values():
			worker.exit()

		# TODO: Get the wt, dt and zt
		return "Done"
	
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
