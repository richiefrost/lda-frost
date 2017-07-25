#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
cimport numpy as np

cdef class CFPlusTree:
	cdef int T
	cdef np.ndarray arr

	# TODO: Buffer syntax for "np.ndarray[DTYPE, ndim=1] arr"
	def __init__(self, np.ndarray arr):
		self.T = arr.shape[0]
		
		self.arr = np.zeros(self.T << 1)
		cdef int idx
		for idx in range((self.T << 1) - 1, 0, -1):
			if idx >= self.T:
				self.arr[idx] = arr[idx - self.T]
			else:
				self.arr[idx] = self.arr[idx << 1] + self.arr[(idx << 1) + 1]

	def get_terms_sum(self):
		return self.arr[1]

	def leaf(self, int leaf_index):
		return self.arr.shape[0] - 1 - self.T + leaf_index

	def leaf_val(self, int leaf_index):
		leaf_index = self.leaf(leaf_index)
		return self.arr[leaf_index]

	def sample(self, double u):
		# Traverse the F+Tree, ending on the sample we want in the leaf node
		cdef int i = 1		
		while i < self.T:
			if u >= self.arr[i << 1]:
				u -= self.arr[i << 1]
				i = (i << 1) + 1
			elif u < self.arr[i << 1]:
				u -= self.arr[(i << 1) + 1]
				i = i << 1
		return i - self.T, self.arr[i]

	def update(self, int leaf_index, double val):
		cdef int i = self.leaf(leaf_index)

		# Go up the tree again, updating sums, starting at the leaf
		while i > 0:
			self.arr[i] += val
			i >>= 1
