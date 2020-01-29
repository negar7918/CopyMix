# MIT License

Copyright (c) 2020 Negar Safinianaini

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This is the implementation of the CopyMix paper available at TODO arxiv:
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
- It takes awhile (recommended to run on a powerful machine or cluster),
  since it is performing VI for 164 initialisation methods each being run for 4 different numbers of clusters in range of 1 to 4.
- The log starts with printing the simulated data; afterwards, you can follow VI iterations, ELBO difference between iterations,
  and the updates of cluster probabilities for each sequence.
- The output is a file to be generated called "inits_45_800_0_result"