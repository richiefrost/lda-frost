import fplus as fp
import data_converter as dc
import rsf_gibbs as rg
import numpy as np
from time import time

def run_loop(num_iter, doc_file, K=10, alpha=0.01, beta=0.01):
		docs = dc.convert_file_to_docs(doc_file)
		vocab = dc.vocab_from_file(doc_file)
		doc_word = dc.convert_to_np(docs, vocab)
		print "%i documents, %i unique words" % doc_word.shape

		topic_counts = [100, 500, 1000]

		print "F+LDA:"
		#for K in topic_counts:
			#t = time()
		sample_fplus(doc_word, K, alpha=alpha, beta=beta)
			#print time() - t

		print "CGS:"
		#for K in topic_counts:
			#t = time()
		sample_fast_cgs(doc_word, K, alpha=alpha, beta=beta)
			#print time() - t

		'''
		dt, wt, zt, WS, DS, ZS = dc.sample_ready(doc_word, K)
		print "Using Gibbs"
		for i in range(num_iter):
			fp.gibbs_sample(WS, DS, ZS, wt, dt, zt, alpha, beta)
			nd = np.sum(dt, axis=1).astype(np.intc)
			ll = rg._loglikelihood(wt, dt, zt, nd, alpha, beta)
			print ll

		'''



def sample_fplus(doc_word, K, alpha=0.01, beta=0.01):
	dt, wt, zt, WS, DS, ZS = dc.sample_ready(doc_word, K)
	randoms = np.array([np.random.random_sample() for i in range(K)])
	for i in range(num_iter):
		fp.sample(WS, DS, ZS, wt, dt, zt, randoms, alpha, beta)
		nd = np.sum(dt, axis=1).astype(np.intc)
		ll = rg._loglikelihood(wt, dt, zt, nd, alpha, beta)
		print ll

def sample_fast_cgs(doc_word, K, alpha=0.01, beta=0.01):
	dt, wt, zt, WS, DS, ZS = dc.sample_ready(doc_word, K)
	rands = np.array([np.random.random_sample() for i in range(K)])
	alphas, betas = np.repeat(alpha, K), np.repeat(beta, K)
	for i in range(num_iter):
		rg._sample_topics(WS, DS, ZS, wt, dt, zt, alphas, betas, rands)
		nd = np.sum(dt, axis=1).astype(np.intc)
		ll = rg._loglikelihood(wt, dt, zt, nd, alpha, beta)
		print ll

if __name__ == "__main__":
	import sys
	doc_file = sys.argv[1]
	num_iter = int(sys.argv[2])
	K = int(sys.argv[3])
	run_loop(num_iter, doc_file, K=K)