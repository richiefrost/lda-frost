import Pyro4
import Pyro4.util
import data_converter as dc
import fplus as fp
import numpy as np
from Queue import Queue, Empty
import sys
import utils

LDA_WORKER_PREFIX = 'lda.worker'
LDA_SERVER_PREFIX = 'lda.server'

class Worker(object):
	def __init__(self):
		sys.excepthook = Pyro4.util.excepthook
		self.doc_word = None
		self.done = False

	@Pyro4.expose
	def initialize(self, dispatcher, server, worker_id, K=10, alpha=0.01, beta=0.01, num_iter=20):
		self.dispatcher = dispatcher
		self.server = server
		self.worker_id = worker_id
		self.K = K
		self.alpha = alpha
		self.beta = beta
		self.num_iter = num_iter
		self.randoms = np.array([np.random.random_sample() for i in range(K)])
		
	
	@Pyro4.expose
	def create_corpus(self):
		self.dt, self.wt, self.zt, self.WS, self.DS, self.ZS = dc.sample_ready(np.array(self.doc_word), self.K)
		self.unique_words = np.unique(self.WS).tolist()
		for i, word in enumerate(self.unique_words):
			indices = [j for j in range(self.wt.shape[1])]
			values = self.wt[word].tolist()
			self.server.update_wt_row(word, indices, values)

	@Pyro4.expose
	def get_dt(self):
		return self.dt.tolist()

	def get_wt(self):
		return np.array([self.server.get_wt_row(word) for word in self.unique_words], dtype=np.intc)

	def get_zt(self):
		return np.array(self.server.get_zt(), dtype=np.intc)

	def send_wt_update(self):
		indices = np.argwhere(self.wt != self.wt_old).tolist()
		values = np.take(self.wt, indices).tolist()
		for i in range(len(indices)):
			word = self.unique_words[indices[i][0]]
			idxs = indices[i]
			vals = values[i]
			self.server.update_wt_row(word, idxs, vals)

	def send_totals_update(self):
		indices = np.where(self.zt != self.zt_old).tolist()
		values = np.take(self.zt, indices).tolist()
		self.server.update_totals(indices, values)

	# Each dw_row is V wide
	@Pyro4.expose
	def receive_dw_row(self, dw_row):
		if self.doc_word is None:
			self.doc_word = np.array(dw_row)
		else:
			self.doc_word = np.append(self.doc_word, np.array(dw_row), axis=0)

	@Pyro4.expose
	@Pyro4.oneway
	def start(self):
		self.create_corpus()
		self.train()

	def train(self):
		for i in range(self.num_iter):
			print "Iteration %i of %i" % (i + 1, self.num_iter)
			self.wt = self.get_wt()
			self.zt = self.get_zt()
			self.wt_old = np.copy(self.wt)
			self.zt_old = np.copy(self.zt)
			fp.sample(self.WS, self.DS, self.ZS, self.wt, self.dt, self.zt, self.randoms, self.alpha, self.beta)
			self.send_wt_update()
			self.send_totals_update()
		self.done = True

	@Pyro4.expose
	def get_done(self):
		return self.done

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
