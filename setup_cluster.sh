#!/bin/bash

if [ $1 == '--help' ] || [ $1 == '-h' ]
	then
	echo "Usage: setup_cluster.sh [--help] [--rebuild-cython]"
	echo "Default tries to install all necessary requirements across nodes, then starts workers on every node."
	echo "Also, the system restarts the Pyro nameserver and the dispatcher."
	echo "The system will kill all workers when restarting, then spawn new workers."
	echo
	echo "--help: Display this message"
	echo "--rebuild-cython: Run default, and rebuild all Cython modules"
	echo "--install-only: Only install unmet requirements on each node, don't start a nameserver, dispatcher or workers"
	exit
fi

if [ $1 != '--install-only' ]
	then
	echo "Restarting nameserver and dispatcher"
	# Reset the nameserver, dispatcher and workers first
	source venv/bin/activate
	python kill_workers.py
	pkill -f pyro4-ns

	# Start the nameserver up again, then dispatcher and workers
	pyro4-ns &
	python dispatcher.py &
fi

while read i
do
	if [ ! -d $i ]
		then
		echo "Directory $i not found, skipping"
		continue
	fi
	cd $i
	# If lda-frost isn't there, we know we need to setup the repo on that cluster
	if [ ! -d lda-frost ]
		then
		echo "Installing on $i"
		git clone https://github.com/richiefrost/lda-frost.git lda-frost
		cd lda-frost
		virtualenv venv
		source venv/bin/activate
		pip install -r requirements.txt
	else
		cd lda-frost
		source venv/bin/activate
	fi
	if [ ! -d build ] || [ $1 == '--rebuild-cython' ]
		then
		echo "Rebuilding Cython on $i"
		bash build.sh
	fi

	if [ $1 != '--install-only' ]
		then
		echo "Starting worker on $i"
		python worker.py &
	fi
done < ./config.cfg