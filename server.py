import numpy as np
import Pyro4
import utils

LDA_SERVER_PREFIX = 'lda.server'

class Server(object):
	@Pyro4.expose
	def initialize(self, V, K):
		self.wt = np.zeros((V, K))
		self.zt = np.zeros(K)
		self.V, self.K = V, K
		
	@Pyro4.expose
	def get_wt_row(self, word):
		return self.wt[word].tolist()

	@Pyro4.expose
	@Pyro4.oneway
	def update_totals(self, indices, values):
		np.put(self.zt, indices, values)

	@Pyro4.expose
	@Pyro4.oneway
	def update_wt_row(self, word, indices, values):
		np.put(self.wt[word], indices, values)
		
	@Pyro4.expose
	def get_zt(self):
		return self.zt.tolist()

	@Pyro4.expose
	def get_wt(self):
		return self.wt.tolist()

	@Pyro4.expose
	@Pyro4.oneway
	def exit(self):
		import os
		print "Shutting down server"
		os._exit(0)

def main():
	utils.pyro_daemon(LDA_SERVER_PREFIX, Server())

if __name__ == "__main__":
	main()
