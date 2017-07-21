import data_converter as dc
import cProfile
from multiprocessing import Pool
import numpy as np
import Pyro4
import Pyro4.util
import sys
from six import iteritems
import utils
from dispatcher import Dispatcher

LDA_DISPATCHER_PREFIX = 'lda.dispatcher'

def run(doc_file, vocab={}, K=10, alpha=0.01, beta=0.01, num_iter=1000):
	if len(vocab) == 0:
		vocab = dc.vocab_from_file(doc_file)
	
	sys.excepthook = Pyro4.util.excepthook

	with utils.getNS() as ns:
		dispatcher = Pyro4.Proxy(ns.list(prefix=LDA_DISPATCHER_PREFIX)[LDA_DISPATCHER_PREFIX])
		dispatcher.initialize(len(vocab), K, alpha, beta, num_iter)

	with open(doc_file) as f:
		for doc in f:
			dw_row = dc.convert_to_np([doc.strip()], vocab)
			dispatcher.add_doc(dw_row.tolist())

	for word in range(len(vocab)):
		token = (word + 1, [0] * K)
		dispatcher.add_initial_token(token)

	# Totals token is reserverd for token index 0
	totals_token = (0, [0] * K)
	dispatcher.add_initial_token(totals_token)

	# Assuming all docs can fit in memory at this point
	dispatcher.train()

	# TODO: Periodically get updates from the workers to report log likelihood

		


if __name__ == '__main__':
	import sys
	doc_file = sys.argv[1]
	run(doc_file)