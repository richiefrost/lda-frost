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
from time import time

LDA_WORKER_PREFIX = 'lda.worker'

class Worker(object):
	def __init__(self):
		sys.excepthook = Pyro4.util.excepthook
		self.doc_word = None
		self.done = False
		self.passed = 0
		self.received = 0
		self.dead_received = 0

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
		
	@Pyro4.expose
	def get_uniques(self):
		return self.uniques

	# zt_global is a "snapshot" of the last time we saw the updated topic totals. We want to update it based on what changed locally
	@Pyro4.expose
	def create_corpus(self):
		self.dt, self.zt, self.WS, self.DS, self.ZS = dc.sample_ready(np.array(self.doc_word), self.K)
		self.uniques = set(np.unique(self.WS))
		self.zt_global = np.copy(self.zt)
		self.iterations_left = {unique: self.num_iter for unique in self.uniques}
		#self.iterations_left[0] = self.num_iter

	@Pyro4.expose
	def create_mappings(self):
		# For each word, list the workers that can use it
		self.token_mappings = {word: [worker_id for worker_id in self.workers if word in self.workers[worker_id].get_uniques()] for word in range(self.V)}

	# Each dw_row is V wide
	@Pyro4.expose
	def receive_dw_row(self, dw_row):
		if self.doc_word is None:
			self.doc_word = np.array(dw_row)
		else:
			self.doc_word = np.append(self.doc_word, np.array(dw_row), axis=0)

	@Pyro4.expose
	def check_done(self):
		return all(self.iterations_left[word] == 0 for word in self.uniques)

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
		self.received += 1
		if incoming_token[0] == 0 or incoming_token[0] - 1 in self.uniques:
			self.job_queue.put(incoming_token)
			token = self.job_queue.get()
			if token[0] > 0 and self.iterations_left[token[0] - 1] <= 0:
				self.dispatcher.remove_worker_mapping(token[0] - 1, self.worker_id)
				self.dead_received += 1
				self.iterations_left[token[0] - 1] = 0
				self.send_token(token)
				return
			if token[0] == 0:
				self.update_totals(token)
			else:
				self.sample(token)
				self.iterations_left[token[0] - 1] -= 1
			self.send_token(token)
		# We have no need for this token, send it along
		else:
			if self.worker_id == 0:
				self.passed += 1
			self.send_token(incoming_token)
			
	@Pyro4.expose
	def tokens_passed(self):
		return self.passed

	@Pyro4.expose
	def tokens_received(self):
		return self.received

	@Pyro4.expose
	def get_worker_id(self):
		return self.worker_id

	@Pyro4.expose
	def get_dead_received(self):
		return self.dead_received

	def send_token(self, token):
		# Only send token to those that can receive it
		if token[0] == 0:
			self.workers[np.random.randint(self.num_workers)].receive_token(token)
		else:
			try:
				mapping = self.dispatcher.get_token_mappings(token[0] - 1)
				if len(mapping) == 0:
					self.dispatcher.receive_finished_token(token)
				else:
					which_worker = mapping[np.random.randint(len(mapping))]
					self.workers[which_worker].receive_token(token)
			except:
				print "Mapping: ", mapping
				print "Length: %i" % len(mapping)

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
