#!/bin/bash

# Reset the nameserver, dispatcher and workers first
source venv/bin/activate
python kill_workers.py
pkill -f pyro4-ns

# Start the nameserver up again, then dispatcher and workers
pyro4-ns &
python dispatcher.py &

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
		git clone https://github.com/richiefrost/lda-frost.git lda-frost
		cd lda-frost
		virtualenv venv
		source venv/bin/activate
		pip install -r requirements.txt
	else
		cd lda-frost
		source venv/bin/activate
	fi
	python worker.py
done < ./config.cfg