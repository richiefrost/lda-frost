# lda-frost - Distributed Latent Dirichlet Allocation with Python and Pyro, using token passing

This project is an experiment to implement text-based LDA with modified token passing as the synchronization technique to keep the word-topic matrix consistent across any number of worker processes.  Worker processes iteratively sample from a subset of the entire corpus when they receive a "token".  Tokens are identified by a 2-tuple, where the first entry in the tuple is the token's ID and the second tuple is a row in the distributed word-topic matrix.  The token ID is the location of a certain word in the vocabulary, and the row is the word-topic row that corresponds to that word.  A dispatcher is used to route token requests amongst workers, and workers can only compute updates to the word-topic matrix for tokens that correspond to words in the word-topic matrix.  Since the model is "owner computes", this model has the potential to be highly parallelizable.

Note: If this is the first time you're setting up the cluster, it'll take awhile to set up, since each node needs to clone this repo, install all Python requirements, etc.  Currently it's only using Python 2, but that's mostly just because the print statements are for Python 2.

## Directions:
1. Create a config.cfg, with each line containing the path to the node you're looking for (again, hopefully this will be available by IP later on)
2. Create a virtual host, with 'venv' as your directory
3. Run source to get into your virtual host environment
4. Install all Python requirements
5. Run setup_cluster.sh
  * If you want to rebuild any Cython modules (the samplers are written in Cython), include the '--rebuild-cython' flag when running setup_cluster.sh


### When you run setup_cluster.sh, this is what happens:
1. Install all the same stuff on every node as you just did in steps 2-4 above
2. Start a Pyro nameserver on the current node
3. Start a dispatcher on the current node (to handle token request routing amongst workers)
4. Start a worker on every node listed in your config.cfg file


### Example (from step 2 on):
```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
bash setup_cluster.sh
```


### Example config.cfg file:
```
/users/Dave/
/users/Charlie/
/users/Mike/
/users/Charlotte/
/users/Betty/
/users/Susie/
```
