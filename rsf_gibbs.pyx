#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

from cython.operator cimport preincrement as inc, predecrement as dec
from libc.stdlib cimport malloc, free
from scipy.special import gammaln

'''
cdef extern from "gamma.h":
	cdef double lda_lgamma(double x) nogil


cdef double lgamma(double x) nogil:
	if x <= 0:
		with gil:
			raise ValueError("x must be strictly positive")
	return lda_lgamma(x)
'''

cdef int searchsorted(double* arr, int length, double value) nogil:
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


def _sample_topics(int[:] WS, int[:] DS, int[:] ZS, int[:, :] nzw, int[:, :] ndz, int[:] nz,
				   double[:] alpha, double[:] beta, double[:] rands):
	cdef int i, k, w, d, z, z_new
	cdef double r, dist_cum
	cdef int N = WS.shape[0]
	cdef int n_rand = rands.shape[0]
	cdef int n_topics = nz.shape[0]
	cdef double beta_sum = 0
	cdef double* dist_sum = <double*> malloc(n_topics * sizeof(double))
	if dist_sum is NULL:
		raise MemoryError("Could not allocate memory during sampling.")
	with nogil:
	
		for i in range(beta.shape[0]):
			beta_sum += beta[i]

		for i in range(N):
			w = WS[i]
			d = DS[i]
			z = ZS[i]

			dec(nzw[w, z])
			dec(ndz[d, z])
			dec(nz[z])

			dist_cum = 0
			for k in range(n_topics):
				# beta is a double so cdivision yields a double
				dist_cum += (nzw[w, k] + beta[w]) / (nz[k] + beta_sum) * (ndz[d, k] + alpha[k])
				dist_sum[k] = dist_cum

			r = rands[i % n_rand] * dist_cum  # dist_cum == dist_sum[-1]
			z_new = searchsorted(dist_sum, n_topics, r)

			ZS[i] = z_new
			inc(nzw[w, z_new])
			inc(ndz[d, z_new])
			inc(nz[z_new])

		free(dist_sum)

cpdef double _loglikelihood(int[:, :] nzw, int[:, :] ndz, int[:] nz, int[:] nd, double alpha, double beta) :
	cdef int k, d
	cdef int D = ndz.shape[0]
	cdef int n_topics = ndz.shape[1]
	cdef int vocab_size = nzw.shape[0]

	cdef double ll = 0

	# calculate log p(w|z)
	cdef double lgamma_beta, lgamma_alpha
	
	lgamma_beta = gammaln(beta)
	lgamma_alpha = gammaln(alpha)

	ll += n_topics * gammaln(beta * vocab_size)
	for k in range(n_topics):
		ll -= gammaln(beta * vocab_size + nz[k])
		for w in range(vocab_size):
			# if nzw[w, k] == 0 addition and subtraction cancel out
			if nzw[w, k] > 0:
				ll += gammaln(beta + nzw[w, k]) - lgamma_beta

	# calculate log p(z)
	for d in range(D):
		ll += (gammaln(alpha * n_topics) -
				gammaln(alpha * n_topics + nd[d]))
		for k in range(n_topics):
			if ndz[d, k] > 0:
				ll += gammaln(alpha + ndz[d, k]) - lgamma_alpha
	return ll
