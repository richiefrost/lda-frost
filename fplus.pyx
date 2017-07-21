#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
cimport numpy as np
from cython.operator cimport preincrement as inc, predecrement as dec
from libc.stdlib cimport malloc, free

def gibbs_sample(WS, DS, ZS, wt, dt, zt, alpha, beta):
	N = WS.shape[0]
	V, K = wt.shape
	prev_doc = -1
	randoms = np.array([np.random.random_sample() for i in range(K)])
	# Initialize the tree for topics only, expecting to add document topic ratios each time we see a new topic
	cumulative = np.zeros(K)
	for i in range(N):
		w = WS[i]
		d = DS[i]
		z = ZS[i]

		wt[w, z] -= 1
		dt[d, z] -= 1
		zt[z] -= 1

		cumsum = 0
		cumulative *= 0
		for t in range(K):
			cumsum += ((wt[w, t] + beta) * (dt[d, t] + alpha) / (zt[t] + V * beta))
			cumulative[t] = cumsum

		uniform = randoms[i % K] * (cumsum)
		z = np.searchsorted(cumulative, uniform)

		ZS[i] = z
		wt[w, z] += 1
		dt[d, z] += 1
		zt[z] += 1

def sample(int[:] WS, int[:] DS, int[:] ZS, int[:] wt, int[:, :] dt, int[:] zt, double[:] randoms, double alpha, double beta, int word):
	cdef int i, t, w, d, z, z_new
	cdef double rT, qT, cumsum, uniform
	cdef int N = WS.shape[0]
	cdef int V = 1 #wt.shape[0]
	cdef int K = wt.shape[0]
	cdef int tree_length = K << 1
	cdef int probs_length = K
	cdef int prev_doc = -1
	cdef double alpha_sum = alpha * V
	cdef double beta_sum = beta * V
	
	cdef double* probs = <double*> malloc(K * sizeof(double))
	if probs is NULL:
		raise MemoryError("Could not allocate probs memory during sampling.")

	for t in range(K):
		probs[t] = (beta * alpha) / (zt[t] + beta_sum)

	cdef double* tree = <double*> malloc((K << 1) * sizeof(double))
	if tree is NULL:
		raise MemoryError("Could not allocate tree memory during sampling.")

	create_tree(probs, tree, K)
	cdef double* rcum = <double*> malloc(K * sizeof(double))
	if rcum is NULL:
		raise MemoryError("Could not allocate rcum memory during sampling.")
	

	for i in range(N):
		w = WS[i]
		d = DS[i]
		z = ZS[i]

		if w != word:
			continue

		dec(wt[z])
		dec(dt[d, z])
		dec(zt[z])

		update_tree(tree, z, tree_length, probs_length, (beta * dt[d, z] / (zt[z] + beta_sum)) - leaf_val(tree, z, tree_length, probs_length))

		# If we have any document values for the previous document, clear em out
		if d != prev_doc and prev_doc != -1:
			for t in range(K):
				update_tree(tree, t, tree_length, probs_length, (-1 * beta * dt[prev_doc, t]) / (zt[t] + beta_sum))

		# Update the fplus tree with this current document's values
		if d != prev_doc:
			for t in range(K):
				update_tree(tree, t, tree_length, probs_length, (beta * dt[d, t]) / (zt[t] + beta_sum))	
				prev_doc = d

		cumsum = 0
		for t in range(K):
			cumsum += (wt[t] * (dt[d, t] + alpha) / (zt[t] + beta_sum))
			rcum[t] = cumsum

		rT = rcum[K-1]
		qT = get_terms_sum(tree)
		uniform = randoms[i % K] * (rT + qT)
		if uniform <= rT:
			z_new = searchsorted(rcum, K, uniform)
		else:
			z_new = tree_sample(tree, uniform - rT, K)

		ZS[i] = z_new
		inc(wt[z_new])
		inc(dt[d, z_new])
		inc(zt[z_new])

		update_tree(tree, z_new, tree_length, probs_length, (beta * dt[d, z_new] / (zt[z_new] + beta_sum)) - leaf_val(tree, z_new, tree_length, probs_length))
	free(rcum)
	free(probs)
	free(tree)


cdef int searchsorted(double* arr, int length, double value):
	"""Bisection search (c.f. numpy.searchsorted)

	Find the index into sorted array `arr` of length `length` such that, if
	`value` were inserted before the index, the order of `arr` would be
	preserved.
	"""
	cdef int imin, imax, imid
	imin = 0
	imax = length
	while imin < imax:
		imid = imin + ((imax - imin) >> 2)
		if value > arr[imid]:
			imin = imid + 1
		else:
			imax = imid
	return imin

'''
TREE STUFF
'''
cdef void create_tree(double* probs, double* tree, int probs_length):
	cdef int idx
	for idx in range((probs_length << 1) - 1, 0, -1):
		if idx >= probs_length:
			tree[idx] = probs[idx - probs_length]
		else:
			tree[idx] = tree[idx << 1] + tree[(idx << 1) + 1]

cdef double get_terms_sum(double* tree):
	return tree[1]

cdef int leaf(int tree_length, int probs_length, int leaf_index):
	return tree_length - 1 - probs_length + leaf_index

cdef double leaf_val(double* tree, int leaf_index, int tree_length, int probs_length):
	leaf_index = leaf(tree_length, probs_length, leaf_index)
	return tree[leaf_index]

cdef int tree_sample(double* tree, double u, int probs_length):
	cdef int i = 1
	while i < probs_length:
		if u >= tree[i << 1]:
			u -= tree[i << 1]
			i = (i << 1) + 1
		else:
			u -= tree[(i << 1) + 1]
			i = i << 1
	return i - probs_length

cdef void update_tree(double* tree, int leaf_index, int tree_length, int probs_length, double value):
	cdef int i = leaf(tree_length, probs_length, leaf_index)

	# Go up the tree again, updating sums, starting at the leaf
	while i > 0:
		tree[i] += value
		i >>= 1
