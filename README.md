# LMM-Lasso-on-large-scale-data

An implementation of the linear mixed model with a Laplacian shrinkage prior (Lasso model) for genotype - phenotype mapping and phenotype prediction, 
which corrects for population strucure. The method LMM-Lasso is explained in: 
- B. Rakitsch, C. Lippert, O. Stegle, K. Borgwardt (2013) A Lasso Multi-Marker Mixed Model for Association Mapping with Population Structure Correction, Bioinformatics 29(2):206-1

New features in this release are: 1. Inclusion of screening rules for faster Lasso solution 2. Adapting scripts to large scale data (like UK Biobank)
The folder AdaScreen corresponds to deployment of the screening rules and the folder LMM-Lasso to data handling for applying the algorithm.

### Usage

Enter in the command line: `python simple_test.py -m 100000 -n 3`
The argument -m regulates the number of samples for which the geno-/phenotype information is read and -n every nth genetic loci from the genotype information file.

### Dependencies
Use python 2.7

- matplotlib
- scipy
- numpy
- sklearn
- pandas
- pysnptools
