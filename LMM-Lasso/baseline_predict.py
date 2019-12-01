import loaddata
import baseline_comparison
import os
import numpy as np
import argparse
import lmm_lasso


def run(mSamples, everynthFeatures, dataset, inpycharm):
    starttime = time.time()
    #name = naming()

    if dataset == "ukbiobank":
        path = ['ukb_chr' + str(chromosome) + '_v2' for chromosome in range(1, 23)] + ['ukb_chrY_v2', 'ukb_chrX_v2', 'ukb_chrXY_v2', 'ukb_chrMT_v2']
        print 'before loading data'
        X, y, name = loaddata.loadukb_server(samples=mSamples, everynthfeature=everynthFeatures, chromosome_path=path)
    else:
        raise NotImplementedError("unknown dataset");

    print 'every %d SNP' % (everynthFeatures)
    print 'the number of samples is', X.shape[0], 'and the number of SNPs is', X.shape[1]
    [n_s, n_f] = X.shape

    y = np.atleast_2d(y).T
    y = lmm_lasso.SNP_standard(y)

    n_train = int(n_s * 0.7)
    n_test = n_s - n_train
    debug = False
    listofname = ["UKbiobank", str(mSamples), "samples", str(n_f), "SNPs"]
    dataset = "".join(listofname)
    #name.dataset = dataset

    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # split into training and testing
    train_idx = np.asarray(range(n_train))
    test_idx = np.asarray(range(n_train, X.shape[0]))

    # lambda path = ...

    # train baseline comparison with lasso
    start1 = time.time()
    res_baseline = baseline_comparison.trainlasso(X[train_idx], X[test_idx], y[train_idx], lambda_path)
    end1 = time.time()
    baseline1 = end1 - start1
    print 'the baseline method finished in %.2fs' % (baseline1)
    res_baseline['predictors'] = np.reshape(res_baseline['predictors'],
                                            (res_baseline['predictors'].shape[0], res_baseline['predictors'].shape[1]))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", help="number of samples", default=1000, type=int)
    parser.add_argument("-n", help="number of nth feature", default=90, type=int)
    parser.add_argument("-d", help="dataset", default="ukbiobank", type=str)
    parser.add_argument("-i", help="inpycharm", default=False, type=bool)
    args = parser.parse_args()
    print 'before in run() function'
    # Call main routine
    run(args.m, args.n, args.d, args.i)
