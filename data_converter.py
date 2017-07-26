import numpy as np
from lda import utils
from collections import defaultdict
import os.path
from six import iteritems

def vocab_from_file(doc_file):
	vocab = set()
	with open(doc_file) as f:
		for line in f:
			[vocab.add(word) for word in line.split()]
	vocab = {word: i for i, word in enumerate(vocab)}
	return vocab

'''
It's assumed that the text, when it gets to this point, has 
already been cleaned, lowercased and lemmatized, etc.
'''
def convert_to_np(docs, vocab):
	#print "Transforming documents"
	D = sum([1 for doc in docs])
	V = len(vocab)
	doc_word = np.zeros((D, V), dtype=np.intc)
	for d, line in enumerate(docs):
		for word in doc.split():
			if word in vocab:
				w = vocab[word]
				doc_word[d, w] += 1
	return doc_word

def convert_file_to_docs(doc_file):
	with open(doc_file) as f:
		docs = [line.strip() for line in f]
	return docs

# wt, dt, zt, WS, DS, ZS
def sample_ready(doc_word, K):
	D, V = doc_word.shape
	dt = np.zeros((D, K), dtype=np.intc)
	wt = np.zeros((V, K), dtype=np.intc)
	zt = np.zeros(K, dtype=np.intc)

	WS, DS = utils.matrix_to_lists(doc_word)
	ZS = np.empty_like(WS, dtype=np.intc)
	N = np.sum(doc_word)
	np.testing.assert_equal(N, len(WS))
	# Randomly assign new topics
	for i in range(N):
		w, d = WS[i], DS[i]
		z_new = np.random.randint(K)
		ZS[i] = z_new
		dt[d, z_new] += 1
		wt[w, z_new] += 1
		zt[z_new] += 1
	return dt, wt, zt, WS, DS, ZS


'''
V = vocab size
K = num topics
D = number of documents
min_L = min document length
max_L = max document length
'''
def generate(V, K, D):
	doc_word = np.random.randint(10, size=(D, V))
	wt, dt, zt, WS, DS, ZS = sample_ready(doc_word, K)
	return doc_word, wt, dt, zt, WS, DS, ZS