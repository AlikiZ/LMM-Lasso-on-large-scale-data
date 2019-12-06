#/usr/bin/python2.7
import matplotlib
matplotlib.use('Agg')
import csv
import scipy as SP
import pdb
import lmm_lasso
import loaddata
import baseline_comparison
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from matplotlib.backends.backend_pdf import PdfPages
from namesforplotting import naming
import time
import argparse
import sys
import resource
import plots
import faulthandler; faulthandler.enable()



def run(mSamples, everynthFeatures, dataset,inpycharm):

    if dataset == "ukbiobank":
        #path = ['ukb_chr' + str(chromosome) + '_v2' for chromosome in range(1, 23)] + ['ukb_chrY_v2', 'ukb_chrX_v2', 'ukb_chrXY_v2', 'ukb_chrMT_v2']
        #X, y, name = loaddata.loadukb_server(samples=mSamples, everynthfeature=everynthFeatures, chromosome_path=path)
        path = ['ukb_chr' + str(chromosome) + '_v2.bed' for chromosome in range(21, 23)]
        X, y, name = loaddata.loadukb_local(samples=mSamples, features=everynthFeatures, chromosome_path=path)
        listofname = ["UKbiobank", str(mSamples), "samples", str(X.shape[1]), "SNPs"]
        name.dataset = "".join(listofname)
    elif dataset == "toydata":
        seed, corr, sigma = 1, 0.7, 0.05
        X, y, w = baseline_comparison.load_toy_data(exms=mSamples, feats=everynthFeatures, seed=seed, corr=corr, sigma=sigma, non_zeros=int(mSamples * 0.2))
        name = naming()
    else:
        raise NotImplementedError("unknown dataset");


    print 'every %d SNP' % (everynthFeatures)
    print 'the number of samples is', X.shape[0], 'and the number of SNPs is', X.shape[1]
    [n_s, n_f] = X.shape

    # standard transformation of target vector y
    y = np.atleast_2d(y).T
    y = lmm_lasso.SNP_standard(y)

    n_train = int(n_s * 0.7)
    n_test = n_s - n_train
    debug = False


    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # train-test split; 3lines for random split and 2lines sequential split
    #train_idx = SP.random.permutation(SP.arange(n_s))
    #test_idx = train_idx[n_train:X.shape[0]]
    #train_idx = train_idx[:n_train]
    train_idx = np.asarray(range(n_train))
    test_idx = np.asarray(range(n_train, X.shape[0]))

    # train LMM-Lasso with and w/o screening rules,
    res = lmm_lasso.train(X[train_idx], y[train_idx], X[test_idx], name, inpycharm, debug=debug)

    w = res['weights']
    w_ada = res['weights_of_adascreen']
    lambda_path = res['lambda_path']
    nz_inds_lasso = res['non-zero indeces lasso']
    nz_inds = res['non-zero indeces ada']
    yhat = res['mean_lasso']
    y_ada = res['mean_ada']

    # train baseline model with standard lasso
    res_baseline = baseline_comparison.trainlasso(X[train_idx], X[test_idx], y[train_idx], lambda_path)
    res_baseline['predictors'] = np.reshape(res_baseline['predictors'], (res_baseline['predictors'].shape[0], res_baseline['predictors'].shape[1]))
    nz_inds_baseline = list(map(lambda x: (x != 0).sum(), res_baseline['weights']))

    name.features(dataset, res['screening_rule'], os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))

    #evaluation metrics
    corr_ada = np.asarray([SP.stats.pearsonr(y_ada[i], y[test_idx])[0] for i in range(y_ada.shape[0])]).flatten()
    corr = np.asarray([SP.stats.pearsonr(yhat[i], y[test_idx])[0] for i in range(yhat.shape[0])]).flatten()
    corr_baseline = np.asarray([SP.stats.pearsonr(res_baseline['predictors'][i,:], y[test_idx].flatten())[0] for i in range(lambda_path.shape[0])])
    ms_error_baseline = np.asarray([mse(res_baseline['predictors'][i,:], y[test_idx].flatten()) for i in range(lambda_path.shape[0])])
    ms_error_ada = np.asarray([mse(y_ada[i].flatten(), y[test_idx].flatten()) for i in range(lambda_path.shape[0])])
    ms_error = np.asarray([mse(yhat[i].flatten(), y[test_idx].flatten()) for i in range(lambda_path.shape[0])])

    maxidx = np.argmax(corr_ada)
    print('... corr(Yhat,Ytrue):{0} for the lambda value {1}'.format(corr[maxidx], lambda_path[maxidx]))
    print('... corr(Yhat,Ytrue): %.2f (in percent)' % (corr[maxidx]))
    print('... corr(Yada,Ytrue): %.2f (in percent)' % (corr_ada[maxidx]))

    # plot
    plots.along_lambdapath(inpycharm, name, lambda_path, corr_ada, corr, corr_baseline, res['lmax'], ms_error, ms_error_ada, ms_error_baseline,
                           nz_inds, nz_inds_lasso, nz_inds_baseline, res['number_screened_f'])

    print 'LMM-Lasso with and without deploying screening rules as well as standard Lasso run'


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", help="number of samples", default=100, type=int)
    parser.add_argument("-n", help="number of nth feature", default=110, type=int)
    parser.add_argument("-d", help="dataset", default="ukbiobank", type=str)
    parser.add_argument("-i", help="inpycharm", default=False, type=bool)
    args = parser.parse_args()

    # Call main routine
    run(args.m, args.n, args.d, args.i)
