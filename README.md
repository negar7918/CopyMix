
This is the implementation of the CopyMix paper:
"CopyMix: Mixture Model Based Single-Cell Clustering and Copy Number Profiling using Variational Inference"

See CopyMix_Gaussian folder for the code and test results. Everything outside this folder is old material when the emission modelling was Poisson.

# Guidelines

- There are 16 simulated data test files (corresponding to the 16 configurations in the paper), and each uses an initialization which results in the largest ELBO. 
- The test results can be reproduced by running each of the file in the CopyMix_Gaussian, where the name starts with "test_".
- By running boxplot.py the test results for the 30 datasets are reproduced in form of a box plot.
- Note that the initialization.py was used to run 50 initializations and provide the one with the largest ELBO.
- For the DLP data, the 50 initializations are saved in the folder 50inits.
- The dlp_analysis.py file can be run (but before that you need to unzip the zipped files) to show the DLP data clustering and copy number inference results when comparing to the original paper.
- The inference.py is the VI code; to run the tests, adjust the pool size in inference.py to match your server's capacity.
- The folder plots provides the plots
- The folder ginkgo provides the clustering comparison of ginkgo and the original paper for DLP data.
- The DLP data files are obtained from https://zenodo.org/record/3445364#.YfPzHvXMLzW

# Guidelines for SNV-CopyMix

- To see the SNV-CopyMix implementation and the test on the DLP data, see folder snv.
- The tensor file in the snv folder contains SNV emissions, covering 14068 sites in the genome. Mutations are treated as "heterozygous". 
- We also tested snv for "homozygous" case; you can change the test in the snv folder by referring, instead, to log_likelihood_tensor_v9.npy
- The two tensor files are obtained by using SCuPhr (https://www.biorxiv.org/content/10.1101/357442v1)
- To run the test in the snv folder, adjust the pool size matching your server. The pool is used to parallelize the inference using multiprocessing in Python.

# Python requirements

- hmmlearn 0.2.1
- matplotlib 3.3.0
- multiprocess 0.70.7
- scikit-learn 0.19.1
- scipy 1.5.4
- tqdm 4.41.0
- pickleshare 0.7.5
- python 3.6.2
  
