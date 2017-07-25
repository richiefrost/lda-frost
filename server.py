import numpy as np
import Pyro4
import rsf_gibbs as sampler
import data_converter as dc

LDA_SERVER_PREFIX = 'lda.server'

@Pyro4.expose
class Server(object):
	def initialize(self, dispatcher):
		self.dispatcher = dispatcher

	def initialize_wt(self, V, K):
		self.wt = np.zeros((V, K))
		self.zt = np.zeros(K)
		self.V, self.K = V, K
		
	def get_wt_row(self, word):
		return self.word_topic[word].tolist()

	def update_totals(self, indices, values):
		np.put(self.zt, indices, values)

	def update_wt_row(self, word, original_word, indices, values):
		np.put(self.word_topic[word], indices, values)
		self.dispatcher.send_token_to_worker(original_word)
		
	def get_zt(self):
		return self.zt.tolist()


def main():
	import random
	with Pyro4.locateNS() as ns:
		with Pyro4.Daemon() as daemon:
			name = '%s.%s' % (LDA_SERVER_PREFIX, hex(random.randint(0, 0xffffff))[2:])
			uri = daemon.register(Server, name)
			ns.register(name, uri)
			print "Server ready at %s" % uri
			daemon.requestLoop()


if __name__ == "__main__":
	main()
