
This is the implementation of the CopyMix paper available at https://www.biorxiv.org/content/10.1101/2020.01.29.926022v2
"CopyMix: Mixture Model Based Single-Cell Clustering and Copy Number Profiling using Variational Inference"


# Generating simulated data

Data simulation steps summary (check file test.py lines 65-93 for more details):

0. Set random seed (for each of 30 datasets, a different seed is set).

1. Generate cells belonging to clusters by sampling from a multinomial distribution given some probability distribution such as uniform for 2 clusters.

2. Generate rates for all cells (rates between 80 to 100); then distribute these rates among the cells for different clusters using step 1.

3. Set values to number of HMM's hidden states, transition matrix of each cluster and sequence length.

4. Generate a Poisson HMM for each cluster using cell rates of that cluster, number of hidden states, transition matrix of the cluster and sequence length;
   this results in one copy number hidden sequence and different cell count sequences emitted from that copy number sequence.

5. Accumulate the cells from step 4, into the dataset. Similarly accumulate their cluster labels and then accumulate the hidden copy number sequences.


# Running simulations

- The test file is an example of the simulation tests (configuration A in Fig. 3) in the paper but for one dataset out of 30.
- The chosen initialization is saved in a file, and, it is enough to run "test.py".
  If you want to rerun the initializations uncomment line 97 in "test.py"; since it takes awhile, it is recommended to run on a powerful machine or cluster.
  The initialization framework performs VI for 164 initialisation methods each being run for 4 different numbers of clusters in range of 1 to 4.

