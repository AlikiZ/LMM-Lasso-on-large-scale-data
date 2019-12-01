"""
lmm_lasso.py

Author:		Barbara Rakitsch
Year:		2012
Group:		Machine Learning and Computational Biology Group (http://webdav.tuebingen.mpg.de/u/karsten/group/)
Institutes:	Max Planck Institute for Developmental Biology and Max Planck Institute for Intelligent Systems (72076 Tuebingen, Germany)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import matplotlib
matplotlib.use('Agg')
import scipy as SP
import scipy.linalg as LA
import scipy.optimize as OPT
import pdb
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import time
import numpy as np
from sklearn import linear_model
import pandas as pd
from pysnptools.snpreader import Bed       #directly reading BED file without csv
#libraries from adascreen
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.getcwd())) + '/AdaScreen'))

#from clustermap import Job, process_jobs
import adascreen.solver #as solver
from sklearn import linear_model
from scripts.experiment_impl import *
from namesforplotting import naming
from matplotlib.backends.backend_pdf import PdfPages
from adascreen.screening_rules import EDPP, DOME, DPP, SAFE, StrongRule, HSConstr
from adascreen.bagscreen import BagScreen
from adascreen.adascreen import AdaScreen
from adascreen.sasvi import Sasvi
from adascreen.solver import *
from adascreen.screening_rules import ScreenDummy
from adascreen.screening_lasso_path import ScreeningLassoPath
from scripts.experiment_view_properties import ExperimentViewProperties
from scripts.experiment_impl import *

from sklearn.utils.extmath import randomized_svd
from plots import plot_speedup
#import primme


def train(X,y, name, inpycharm, numintervals=100,ldeltamin=-5,ldeltamax=5,rho=1,alpha=1,debug=False):
    """ K
    train linear mixed model lasso

    Input:
    X: SNP matrix: n_s x n_f
    y: phenotype:  n_s x 1
    Xtest: SNP matrix of the test IDs
    K: kinship matrix: n_s x n_s
    mu: l1-penalty parameter
    numintervals: number of intervals for delta linesearch
    ldeltamin: minimal delta value (log-space)
    ldeltamax: maximal delta value (log-space)
    rho: augmented Lagrangian parameter for Lasso solver
    alpha: over-relatation parameter (typically ranges between 1.0 and 1.8) for Lasso solver

    Output:
    results
    """
    print 'train LMM-Lasso'

    [n_s,n_f] = X.shape
    assert X.shape[0]==y.shape[0], 'dimensions do not match'
    if y.ndim==1:
        y = SP.reshape(y,(n_s,1))

    # train null model
    S,U,ldelta0 = train_nullmodel(y,X[:, ::6],numintervals,ldeltamin,ldeltamax,debug=debug)

    # transform data
    delta0 = SP.exp(ldelta0)
    Sdi = 1. / (np.power(S, 2) + delta0)  # == 1/(S+delta0), via SVD I have sqrt(eigenval) = singularval
    Sdi_sqrt = SP.sqrt(Sdi)
    SUX = SP.dot(U.T, X)
    X2 = SP.sqrt(1. / delta0) * (X - SP.dot(U, SUX))
    SUX = np.vstack((SP.tile(Sdi_sqrt, (SUX.shape[1], 1)).T * SUX, X2))  # SP.dot(SP.tile(Sdi_sqrt, (len(S), 1)).T
    SUy = SP.dot(U.T, y)
    Y2 = SP.sqrt(1. / delta0) * (y - SP.dot(U, SUy))
    SUy = np.vstack((SUy * SP.reshape(Sdi_sqrt, (len(S), 1)), Y2))

    # train lasso on residuals
    beta, nz_inds, scr_inds, path, screening_rule, times_solver, lmax, solver = train_lasso_5(SUX, SUy)

    paok = map(lambda x: train_lasso_sklearn(SUX, SUy, x), path)
    weights = np.asarray([p[0] for p in paok]).T
    time_lasso = np.asarray([p[1] for p in paok])
    nzlasso = np.asarray([p[2] for p in paok])

    # calculate the mean of the Gaussian predictive distribution
    Ktt_hat = 1. / n_f * SP.dot(SUX, SUX.T)
    Ktt_hat = SP.dot(SUX.T, np.linalg.inv(Ktt_hat + delta0 * np.identity(Ktt_hat.shape[0])))
    mean_ada = np.asarray([predict_mean(Ktt_hat, SUy - SP.dot(SUX, np.reshape(beta[:, i], (beta.shape[0], 1))), beta[:, i], Xtest) for i in range(beta.shape[1])])
    mean_lasso = np.asarray([predict_mean(Ktt_hat, SUy - SP.dot(SUX, np.reshape(weights[:, i], (weights.shape[0],1) )), weights[:, i], Xtest) for i in range(weights.shape[1])] )
    
    #def predict_phenotype(SUX, n_f, delta0, SUy, beta, Xtest, weights):
    #    return mean_ada, mean_lasso

    plot_speedup(SUX, time_lasso, name, lmax, solver, inpycharm, screening_rules=screening_rule, path=path, times_solver=times_solver)

    if beta.shape == weights.T.shape:
        beta = beta.T

    amountof_screened_feat = list(map(lambda x: len(x), scr_inds))
    nz_inds = list(map(lambda x: len(x), nz_inds))


    solver_gap = list(map(lambda i: lasso_objective(SUy, SUX, weights[:,i], path[i]) - lasso_objective(SUy, SUX, beta[:,i], path[i]), range(len(path))))
    print solver_gap

    res = {}
    res['ldelta0'] = ldelta0
    res['weights'] = weights
    res['time'] = time_diff
    res['weights_of_adascreen'] = beta
    res['lambda_path'] = path
    res['number_screened_f'] = amountof_screened_feat
    res['time_solutions_admm'] = time_lasso
    res['screening_rule'] = screening_rule
    res['lmax'] =lmax
    res['non-zero indeces ada'] =nz_inds
    res['non-zero indeces lasso'] = nzlasso
    res['mean_ada'] = np.reshape(mean_ada, (mean_ada.shape[0],mean_ada.shape[1]))
    res['mean_lasso'] = np.reshape(mean_lasso, (mean_lasso.shape[0],mean_lasso.shape[1]))
    return res


def train_lasso_sklearn(X,y,mu, zero_threshold=1E-3):
    startTime = time.time()
    model = linear_model.Lasso(alpha=mu, fit_intercept=False)
    model.fit(X * np.sqrt(X.shape[0]), y * np.sqrt(X.shape[0]))
    model.coef_[SP.absolute(model.coef_)<zero_threshold]=0
    totaltime = time.time() - startTime
    nonzero = (model.coef_ != 0).sum()
    return model.coef_, totaltime, nonzero


def lasso_objective(y, x, weights, mu):
    if weights.ndim != 2:
        weights = weights.reshape(weights.shape[0], 1)
    return 0.5*np.sum((y - np.dot(x,weights))**2) + mu*np.sum(np.abs(weights))


def predict(y_train,X_train,X_v,ldelta,K_tt, K_vt,w):
    """ K_tt,K_vt,
    predict the phenotype

    Input:
    y_train: phenotype: n_train x 1
    X_train: SNP matrix: n_train x n_f
    X_v: SNP matrix: n_val x n_f
    ldelta: kernel parameter
    K_tt: kinship matrix: n_train x n_train
    K_vt: kinship matrix: n_val  x n_train
    w: lasso weights: n_f x 1

    Output:
    y_v: predicted phenotype: n_val x 1
    """
    [n_train, n_f] = X_train.shape
    n_test = X_v.shape[0]

    #K_tt = 1. / n_f(this should be the initial n_f) * SP.dot(X_train, X_train.T)
    #K_vt = 1. / n_f(this should be the initial n_f) * SP.dot(X_v, X_train.T)
    if y_train.ndim==1:
        y_train = SP.reshape(y_train,(n_train,1))
    if w.ndim==1:
        w = SP.reshape(w,(n_f,1))

    assert y_train.shape[0]==X_train.shape[0], 'dimensions do not match'
    assert y_train.shape[0]==K_tt.shape[0], 'dimensions do not match'
    assert y_train.shape[0]==K_tt.shape[1], 'dimensions do not match'
    assert y_train.shape[0]==K_vt.shape[1], 'dimensions do not match'
    assert X_v.shape[0]==K_vt.shape[0], 'dimensions do not match'
    assert X_train.shape[1]==X_v.shape[1], 'dimensions do not match'
    assert X_train.shape[1]==w.shape[0], 'dimensions do not match'

    
    delta = SP.exp(ldelta)
    idx = w.nonzero()[0]

    if idx.shape[0]==0:
        return SP.dot(K_vt,LA.solve(K_tt + delta*SP.eye(n_train),y_train))
        
    y_v = SP.dot(X_v[:,idx],w[idx]) + SP.dot(K_vt, LA.solve(K_tt + delta*SP.eye(n_train),y_train-SP.dot(X_train[:,idx],w[idx])))

    return y_v


"""
helper functions
"""

def predict_mean(a, b, weight, Xtest):
    """
    a: SUX.T* (SUX*SUX.T + delta*I)^(-1)
    b: SUy - SUX*weight
    weight: weight of the lasso for a fixed lambda
    result: Xtest * (a*b + weight)
    """
    return SP.dot(Xtest, (1. / Xtest.shape[1])* (1. / Xtest.shape[1]) * SP.dot(a, b) + np.reshape(weight, (a.shape[0], b.shape[1])) )

def normalize(z):
    return (z - min(z)) / (max(z) - min(z))

def SNP_standard(X):
    #return np.array(map(lambda j: (X[:, j] - X[:, j].mean()) / X[:, j].std(), range(X.shape[1]) )).T
    return  np.array([(X[:, j] - X[:, j].mean()) / X[:, j].std() if X[:, j].std() != 0 else X[:,j] for j in range(X.shape[1])]).T


def train_lasso_5(X, y, geomul=0.9, lower_bound=0.001, steps=40):
    solvers = [SklearnCDSolver(), SklearnLarsSolver(),
              ProximalGradientSolver(), AccelProximalGradientSolver()]   #ActiveSetCDSolver, GlmnetSolver
    screening_rls = DPP()
    solver = solvers[0]
    myLasso = ScreeningLassoPath(screening_rls, solver, path_lb=lower_bound, path_steps=steps, path_stepsize=geomul,
                                path_scale='linear')
    beta, nz_inds, scr_inds, path, times_solver, times_screening, lmax = myLasso.fit(X.T, y, max_iter=1000, tol=1e-4, debug=False)
    print "the greatest value of the lambda path", lmax
    return beta, nz_inds, scr_inds, path, screening_rls, times_solver, lmax, solver


def nLLeval(ldelta,Uy,S):
    """
    evaluate the negative log likelihood of a random effects model:
    nLL = 1/2(n_s*log(2pi) + logdet(K) + 1/ss * y^T(K + deltaI)^{-1}y,
    where K = USU^T.

    Uy: transformed outcome: n_s x 1
    S:  eigenvectors of K: n_s
    ldelta: log-transformed ratio sigma_gg/sigma_ee
    """
    n_s = Uy.shape[0]
    delta = SP.exp(ldelta)
    
    # evaluate log determinant
    Sd = S+delta
    ldet = SP.sum(SP.log(Sd))

    # evaluate the variance    
    Sdi = 1.0/Sd
    Uy = Uy.flatten()
    ss = 1./n_s * (Uy*Uy*Sdi).sum()

    # evalue the negative log likelihood
    nLL=0.5*(n_s*SP.log(2.0*SP.pi)+ldet+n_s+n_s*SP.log(ss));

    return nLL


def train_nullmodel(y,X_subsampled,numintervals=100,ldeltamin=-5,ldeltamax=5,debug=True):
    """
    train random effects model:
    min_{delta}  1/2(n_s*log(2pi) + logdet(K) + 1/ss * y^T(K + deltaI)^{-1}y,
    
    Input:
    X: SNP matrix: n_s x n_f
    y: phenotype:  n_s x 1
    X_subsampled: design matrix of SNPs
    K: kinship matrix: n_s x n_s    #not used at the moment
    mu: l1-penalty parameter
    numintervals: number of intervals for delta linesearch
    ldeltamin: minimal delta value (log-space)
    ldeltamax: maximal delta value (log-space)
    """
    if debug:
        print '... train null model'
        
    n_s = y.shape[0]
    n_f = X_subsampled.shape[1]

    # rotate data
    U, S, _ = SP.linalg.svd(np.sqrt(1. / n_f) * X_subsampled, lapack_driver='gesvd', overwrite_a=True, full_matrices=False)
    #U, S, _ = randomized_svd(np.sqrt(1. / n_f)*X_subsampled, n_components= int(min(X_subsampled.shape) * 0.7))

    Uy = SP.dot(U.T,y)
    # grid search
    nllgrid=SP.ones(numintervals+1)*SP.inf
    ldeltagrid=SP.arange(numintervals+1)/(numintervals*1.0)*(ldeltamax-ldeltamin)+ldeltamin
    nllmin=SP.inf
    for i in SP.arange(numintervals+1):
        nllgrid[i]=nLLeval(ldeltagrid[i],Uy,S*S);     #S_eigenval = S_SVD * S_SVD
    # find minimum
    ldeltaopt_glob = ldeltagrid[nllgrid.argmin()]

    # more accurate search around the minimum of the grid search
    for i in SP.arange(numintervals-1)+1:
        if (nllgrid[i]<nllgrid[i-1] and nllgrid[i]<nllgrid[i+1]):
            ldeltaopt,nllopt,iter,funcalls = OPT.brent(nLLeval,(Uy,S*S),(ldeltagrid[i-1],ldeltagrid[i],ldeltagrid[i+1]),full_output=True);
            if nllopt<nllmin:
                nllmin=nllopt;
                ldeltaopt_glob=ldeltaopt;

    print 'the null model function'

    return S,U,ldeltaopt_glob


def leastFrequent(arr):
    n = len(arr)
    # Insert all elements in Hash.
    Hash = dict()
    for i in range(n):
        if arr[i] in Hash.keys():
            Hash[arr[i]] += 1
        else:
            Hash[arr[i]] = 1

    # find the min absolute frequency
    min_count = min(Hash.values())
    return min_count


def lasso_obj(X,y,w,mu,z):
    """
    evaluates lasso objective: 0.5*sum((y-Xw)**2) + mu*|z|

    Input:
    X: design matrix: n_s x n_f
    y: outcome:  n_s x 1
    mu: l1-penalty parameter
    w: weights: n_f x 1
    z: slack variables: n_fx1

    Output:
    obj
    """
    return 0.5*((SP.dot(X,w)-y)**2).sum() + mu*SP.absolute(z).sum()