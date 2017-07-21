import Pyro4
import Pyro4.util
import data_converter as dc
import fplus as fp
import numpy as np
from Queue import Queue
import threading
import sys
import utils

LDA_WORKER_PREFIX = 'lda.worker'

class Worker(object):
	def __init__(self):
		sys.excepthook = Pyro4.util.excepthook
		self.doc_word = None
		self.done = False

	@Pyro4.expose
	def initialize(self, dispatcher, worker_id, K=10, alpha=0.01, beta=0.01, num_iter=20):
		self.lock_update = threading.Lock()
		self.randoms = np.array([np.random.random_sample() for i in range(K)])
		self.dispatcher = dispatcher
		self.worker_id = worker_id
		self.K = K
		self.alpha = alpha
		self.beta = beta
		self.num_iter = num_iter
		self.job_queue = Queue()

	# zt_global is a "snapshot" of the last time we saw the updated topic totals. We want to update it based on what changed locally
	@Pyro4.expose
	def create_corpus(self):
		self.dt, self.zt, self.WS, self.DS, self.ZS = dc.sample_ready(np.array(self.doc_word), self.K)
		self.zt_global = np.copy(self.zt)
		uniques = np.unique(self.WS)
		self.iterations_left = {unique: self.num_iter for unique in uniques}
		self.unique_words = set(uniques)

	# Each dw_row is V wide
	@Pyro4.expose
	def receive_dw_row(self, dw_row):
		if self.doc_word is None:
			self.doc_word = np.array(dw_row)
		else:
			self.doc_word = np.append(self.doc_word, np.array(dw_row), axis=0)

	def can_receive_token(self, token):
		word = token[0] - 1
		if word == -1:
			return True
		if word not in self.unique_words:
			return False
		if self.iterations_left[word] <= 0:
			return False
		return True

	@Pyro4.expose
	def check_done(self):
		return all(self.iterations_left[word] == 0 for word in self.unique_words)

	def update_totals(self, token):
		s = np.array(token[1], dtype=np.intc)
		s_ = self.zt_global
		sl = self.zt
		s = s + sl - s_
		self.zt_global = s
		self.zt = s
		token = (0, s.tolist())

	@Pyro4.expose
	@Pyro4.oneway
	def request_token(self):
		if self.dispatcher is None:
			raise RuntimeError("Worker must be initialized before requesting tokens")

		token = None
		while token is None and not self.done:
			try:
				token = self.dispatcher.get_token()
			except Queue.Empty:
				# No tokens currently available. See if we're done, or keep trying
				if self.check_done():
					print "Done processing on worker %i" % self.worker_id
					self.done = True
					self.exit()
				continue
		if token is not None:
			if token[0] == 0:
				self.update_totals(token)
				self.dispatcher.receive_token(token, self.worker_id)
			else:
				if self.can_receive_token(token):
					self.sample(token)
					self.iterations_left[token[0] - 1] -= 1
				self.dispatcher.receive_token(token, self.worker_id)
		
			
	@Pyro4.expose
	def start(self):
		self.create_corpus()

	#@utils.synchronous('lock_update')
	def sample(self, token):
		word = token[0] - 1
		row = np.array(token[1], dtype=np.intc)
		fp.sample(self.WS, self.DS, self.ZS, row, self.dt, self.zt, self.randoms, self.alpha, self.beta, word)
		token = (word + 1, row.tolist())

	@Pyro4.expose
	@Pyro4.oneway
	def exit(self):
		import os
		print "Shutting down worker %i" % self.worker_id
		os._exit(0)

def main():
	import random
	with Pyro4.locateNS() as ns:
		with Pyro4.Daemon() as daemon:
			name = '%s.%s' % (LDA_WORKER_PREFIX, hex(random.randint(0, 0xffffff))[2:])
			uri = daemon.register(Worker(), name)
			ns.remove(name)
			ns.register(name, uri)
			print "Worker ready at %s" % uri
			daemon.requestLoop()


if __name__ == "__main__":
	main()
