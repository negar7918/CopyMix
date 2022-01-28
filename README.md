
This is the implementation of the CopyMix paper available at https://www.biorxiv.org/content/10.1101/2020.01.29.926022v2
"CopyMix: Mixture Model Based Single-Cell Clustering and Copy Number Profiling using Variational Inference"

See CopyMix_Gaussian folder for the code and test results. Everything outside this folder is old material when the emission modelling was Poisson.

# Guidelines

- There are 18 simulated data test files (corresponding to the 18 configurations in the paper), and each uses an initialization which results in the largest ELBO. 
- The initialization.py was used, for each test, to run 50 initializations and provide the one with the largest ELBO.
- For the DLP data, the 50 initializations are saved in folder 50inits.
- The dlp_analysis.py file can  be run to show the DLP data clustering and copy number inference results when comparing to the original paper.
- The inference.py is the VI code.
- The folder ginkgo provides the clustering comparison of ginkgo and the original paper for DLP data.

# Running requirements

- hmmlearn 0.2.1
- matplotlib 3.3.0
- multiprocess 0.70.7
- scikit-learn 0.19.1
- scipy 1.5.4
- tqdm 4.41.0
- pickleshare 0.7.5
  