# lda-frost - distributed Latent Dirichlet Allocation with Python and Pyro
To run this on my cluster, I'm able to reach each machine by an SSH session to a single node, then cd'ing into each individual node via bash.  To make this more robust, it'll need to be able to handle IP addresses rather than directories.

Note: If this is the first time you're setting up the cluster, it'll take awhile to set up, since each node needs to clone this repo, install all Python requirements, etc.  Currently it's only using Python 2, but that's mostly just because the print statements are for Python 2.

Directions:
1. Create a config.cfg, with each line containing the path to the node you're looking for (again, hopefully this will be available by IP later on)
2. Create a virtual host, with 'venv' as your directory
3. Run source to get into your virtual host environment
4. Install all Python requirements
5. Run setup_cluster.sh

When you run setup_cluster.sh, you're essentially doing the following:
1. installing all the same stuff on every node as you just did in steps 2-4
2. Starting a Pyro nameserver on the current node
3. Starting a dispatcher on the current node (to handle token request routing amongst workers)
4. Starting a worker on every node listed in your config.cfg file

Example (from step 2 on):
```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
bash setup_cluster.sh
```

Example config.cfg file:
```
/users/Dave/
/users/Charlie/
/users/Mike/
/users/Charlotte/
/users/Betty/
/users/Susie/
```
