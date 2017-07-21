import random
import logging
import Pyro4
from Queue import Queue
import data_converter as dc
import numpy as np
import fplus as fp
import rsf_gibbs as rg
from time import time
from functools import wraps  # for `synchronous` function lock

logger = logging.getLogger(__name__)

def run(doc_file, vocab_file, K=10, alpha=0.01, beta=0.01, num_iter=20):
	vocab = dc.vocab_from_file(vocab_file)
	# Word-topic + doc-topic should be less than 30 MB
	V = len(vocab)
	# Try to get the number of documents processed at a time to stay in the L3 cache (play around with this and see what works)
	batch_size = 5 * 10**8 / V
	#print "Running batch size of %i" % batch_size
	#print "K = %i, V = %i" % (K, V)
	queue = Queue(maxsize=batch_size)
	global_nd = np.zeros(batch_size, dtype=np.intc)
	global_wt = np.zeros((V, K), dtype=np.intc)
	global_zt = np.zeros(K, dtype=np.intc)

	t = time()
	with open(doc_file) as f:
		for j, line in enumerate(f):
			if not queue.full():
				queue.put(line.strip())
			else:
				#print "Queue full, starting to process a batch"
				docs = []
				while not queue.empty():
					doc = queue.get()
					docs.append(doc)
				
				doc_word = dc.convert_to_np(docs, vocab)
				docs = []
				dt, wt, zt, WS, DS, ZS = dc.sample_ready(doc_word, K)
				del doc_word
				# Update the local wt and zt with the latest global versions
				wt += global_wt
				zt += global_zt
				randoms = np.array([np.random.random_sample() for i in range(K)])
				#print "Starting another batch"
				for i in range(num_iter):
					fp.sample(WS, DS, ZS, wt, dt, zt, randoms, alpha, beta)
					nd = np.sum(dt, axis=1).astype(np.intc)
					ll = rg._loglikelihood(wt, dt, zt, nd, alpha, beta)
					#print ll

				# Update the globals
				#global_nd = nd
				global_wt = wt
				global_zt = zt
				print "Processed %i so far" % j
	print "Final log likelihood: %d" % ll
	print time() - t



def pyro_daemon(name, obj, random_suffix=False, ip=None, port=None, ns_conf={}):
	"""
	Register object with name server (starting the name server if not running
	yet) and block until the daemon is terminated. The object is registered under
	`name`, or `name`+ some random suffix if `random_suffix` is set.

	"""
	if random_suffix:
		name += '.' + hex(random.randint(0, 0xffffff))[2:]
	import Pyro4
	with getNS(**ns_conf) as ns:
		with Pyro4.Daemon(ip or get_my_ip(), port or 0) as daemon:
			# register server for remote access
			uri = daemon.register(obj, name)
			ns.remove(name)
			ns.register(name, uri)
			logger.info("%s registered with nameserver (URI '%s')" % (name, uri))
			daemon.requestLoop()

# From gensim
def getNS(host=None, port=None, broadcast=True, hmac_key=None):
	"""
	Return a Pyro name server proxy.
	"""
	try:
		return Pyro4.locateNS(host, port, broadcast, hmac_key)
	except Pyro4.errors.NamingError:
		raise RuntimeError("Pyro name server not found")

def get_my_ip():
	"""
	Try to obtain our external ip (from the pyro nameserver's point of view)

	This tries to sidestep the issue of bogus `/etc/hosts` entries and other
	local misconfigurations, which often mess up hostname resolution.

	If all else fails, fall back to simple `socket.gethostbyname()` lookup.

	"""
	import socket
	try:
		from Pyro4 import naming
		# we know the nameserver must exist, so use it as our anchor point
		ns = naming.locateNS()
		s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		s.connect((ns._pyroUri.host, ns._pyroUri.port))
		result, port = s.getsockname()
	except:
		try:
			# see what ifconfig says about our default interface
			import commands
			result = commands.getoutput("ifconfig").split("\n")[1].split()[1][5:]
			if len(result.split('.')) != 4:
				raise Exception()
		except:
			# give up, leave the resolution to gethostbyname
			result = socket.gethostbyname(socket.gethostname())
	return result


def synchronous(tlockname):
    """
    A decorator to place an instance-based lock around a method.

    Adapted from http://code.activestate.com/recipes/577105-synchronization-decorator-for-class-methods/
    """
    def _synched(func):
        @wraps(func)
        def _synchronizer(self, *args, **kwargs):
            tlock = getattr(self, tlockname)
            logger.debug("acquiring lock %r for %s" % (tlockname, func.__name__))

            with tlock:  # use lock as a context manager to perform safe acquire/release pairs
                logger.debug("acquired lock %r for %s" % (tlockname, func.__name__))
                result = func(self, *args, **kwargs)
                logger.debug("releasing lock %r for %s" % (tlockname, func.__name__))
                return result
        return _synchronizer
    return _synched