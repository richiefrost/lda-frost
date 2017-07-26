#!/bin/bash
source venv/bin/activate
python kill_workers.py
python worker.py &
python server.py &
python dispatcher.py &
python worker.py &
python driver.py data/coldplay.txt