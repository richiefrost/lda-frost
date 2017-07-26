import Pyro4
import utils
from six import iteritems

LDA_WORKER_PREFIX = 'lda.worker'
LDA_DISPATCHER_PREFIX = 'lda.dispatcher'
LDA_SERVER_PREFIX = 'lda.server'

with utils.getNS() as ns:
	for name, uri in iteritems(ns.list(prefix=LDA_WORKER_PREFIX)):
		try:
			worker = Pyro4.Proxy(uri)
			worker.exit()
			ns.remove(name)
		except Pyro4.errors.PyroError:
			print "Error trying to kill worker at %s" % uri
			ns.remove(name)
	dispatcher_list = ns.list(prefix=LDA_DISPATCHER_PREFIX)
	if LDA_DISPATCHER_PREFIX in dispatcher_list:
		dispatcher = Pyro4.Proxy(dispatcher_list[LDA_DISPATCHER_PREFIX])
		dispatcher.exit()
		ns.remove(LDA_DISPATCHER_PREFIX)

	server_list = ns.list(prefix=LDA_SERVER_PREFIX)
	if LDA_SERVER_PREFIX in server_list:
		server = Pyro4.Proxy(server_list[LDA_SERVER_PREFIX])
		server.exit()
		ns.remove(LDA_SERVER_PREFIX)