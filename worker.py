import Pyro4
import Pyro4.util
import data_converter as dc
import fplus as fp
import numpy as np
from Queue import Queue
from Queue import Empty
import threading
import sys
import utils
from six import iteritems

LDA_WORKER_PREFIX = 'lda.worker'

class Worker(object):
	def __init__(self):
		sys.excepthook = Pyro4.util.excepthook
		self.doc_word = None
		self.done = False

	@Pyro4.expose
	def initialize(self, dispatcher, worker_id, K=10, V=100, alpha=0.01, beta=0.01, num_iter=20):
		self.randoms = np.array([np.random.random_sample() for i in range(K)])
		self.dispatcher = dispatcher
		self.worker_id = worker_id
		self.K = K
		self.V = V
		self.alpha = alpha
		self.beta = beta
		self.num_iter = num_iter
		self.job_queue = Queue()

	@Pyro4.expose
	def include_workers(self, workers):
		self.workers = workers
		self.num_workers = len(workers)

	# zt_global is a "snapshot" of the last time we saw the updated topic totals. We want to update it based on what changed locally
	@Pyro4.expose
	def create_corpus(self):
		self.dt, self.zt, self.WS, self.DS, self.ZS = dc.sample_ready(np.array(self.doc_word), self.K)
		self.zt_global = np.copy(self.zt)
		self.iterations_left = {i: self.num_iter for i in range(self.V + 1)}

	# Each dw_row is V wide
	@Pyro4.expose
	def receive_dw_row(self, dw_row):
		if self.doc_word is None:
			self.doc_word = np.array(dw_row)
		else:
			self.doc_word = np.append(self.doc_word, np.array(dw_row), axis=0)

	@Pyro4.expose
	def check_done(self):
		return all(self.iterations_left[word] == 0 for word in range(self.K))

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
	def receive_token(self, incoming_token):
		self.job_queue.put(incoming_token, block=True)
		token = self.job_queue.get()
		if self.iterations_left[token[0]] <= 0:
			self.iterations_left[token[0]] = 0
			self.send_token(token)
			return
		if token[0] == 0:
			self.update_totals(token)
		else:
			self.sample(token)
		self.iterations_left[token[0]] -= 1
		self.send_token(token)

	def send_token(self, token):
		self.workers[np.random.randint(self.num_workers)].receive_token(token)

	#@utils.synchronous('lock_update')
	def sample(self, token):
		word = token[0] - 1
		row = np.array(token[1], dtype=np.intc)
		fp.sample(self.WS, self.DS, self.ZS, row, self.dt, self.zt, self.randoms, self.alpha, self.beta, word)
		token = (token[0], row.tolist())

	@Pyro4.expose
	@Pyro4.oneway
	def exit(self):
		import os
		print "Shutting down worker"
		os._exit(0)

def main():
	utils.pyro_daemon(LDA_WORKER_PREFIX, Worker(), random_suffix=True)

if __name__ == "__main__":
	main()
