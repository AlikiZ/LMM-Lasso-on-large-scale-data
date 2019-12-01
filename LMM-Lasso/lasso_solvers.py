import scipy.linalg
import numpy as np
import time
import scipy
import argparse
import baseline_comparison
import loaddata


def load_toy_data(exms=100, feats=10000, seed=None, sigma=0.0, corr=0.0, non_zeros=100):
    """
    data generation, code taken from Adascreen implementation.

    Returns:
        X: predictor data
        y: regression targets
        beta_star: true regression coefficients that were used to generate the data
    """
    if seed != None:
        np.random.seed(seed)
    # Generate data similar as done in the Sasvi paper
    X = np.random.uniform(low=0., high=+1., size=(exms, feats))
    for i in range(1, feats):
        X[:, i] = (1.0 - corr) * X[:, i] + corr * X[:, i - 1]
    # (exms x features)
    beta_star = np.random.uniform(low=-1., high=+1., size=feats)
    # set 'cut' number of coefficients to zero
    cut = feats - non_zeros
    # random shuffling of indices
    inds = np.random.permutation(range(feats))
    # set 'cut' random coefficients to zero
    beta_star[inds[:cut]] = 0.0
    # compue the noisy responses for the regression
    y = X.dot(beta_star) + sigma * np.random.rand(exms)
    return X, y, beta_star

def run(mSamples, nFeatures):
    """
    Main routine: generates data, times the Lasso solvers
    Args:
    ----
        mSamples: number of observations in generated training data
        nFeatures: number of features in generated training data
    from scikit learn: To avoid unnecessary memory duplication the X argument of the fit method
     should be directly passed as a fortran contiguous numpy array
    """

    #X = np.random.rand(mSamples, nFeatures)
    #seed, corr, sigma = 1, 0.7, 0.05
    #X, y, w = load_toy_data(exms=mSamples, feats=nFeatures, seed=seed, corr=corr, sigma=sigma,non_zeros=int(nFeatures * 0.2))
    path = ['ukb_chr' + str(chromosome) + '_v2' for chromosome in range(1, 23)] + ['ukb_chrY_v2', 'ukb_chrX_v2', 'ukb_chrXY_v2', 'ukb_chrMT_v2']
    print 'before loading data'
    X, y, name = loaddata.loadukb_server(samples=mSamples, everynthfeature=everynthFeatures, chromosome_path=path)

    mu = 10

    glmnet = []
    LassoLars = []
    Lasso = []

    for i in range(10):
        start1 = time.time()
        weight = baseline_comparison.train_lasso_3(X, y, mu)        #glmnet
        end1 = time.time()
        timeglmnet = end1 - start1
        glmnet.append(timeglmnet)
        print 'the glmnet finished in %.5fs' % (timeglmnet)

        start1 = time.time()
        weight = baseline_comparison.train_lasso_3(X, y, mu)        #.LassoLars
        end1 = time.time()
        timeLassoLars = end1 - start1
        LassoLars.append(timeLassoLars)
        print 'the linear_model.LassoLars finished in %.5fs' % (timeLassoLars)

        start1 = time.time()
        weight = baseline_comparison.train_lasso_2(X, y, mu)        #.Lasso
        end1 = time.time()
        timeLasso = end1 - start1
        Lasso.append(timeLasso)
        print 'the linear_model.Lasso finished in %.5fs' % (timeLasso)

    print 'the glmnet', glmnet
    print 'the linear_model.LassoLars', LassoLars
    print 'the linear_model.LassoLars', Lasso



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", help="number of samples", default=640 / 2, type=int)
    parser.add_argument("-n", help="number of features", default=10, type=int)
    args = parser.parse_args()

    # Call main routine
    run(args.m, args.n)
