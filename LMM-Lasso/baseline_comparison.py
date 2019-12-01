import numpy as np
import lmm_lasso
from sklearn import linear_model

"""
X, y, _ = lmm_lasso.load_toy_data()

path = np.random.rand(3,1)
hh = train_lasso_2(X, y, path)
r = trainlasso(X, y, path)
  """

#output: get weights along the lambda path, predict yhat, make the plots
#do it similar to lmm_lasso
def trainlasso(Xtrain, Xtest, ytrain, path):
    weights = np.array(map(lambda x: train_lasso_2(Xtrain, ytrain, x), path))
    y_hat = np.array(map(lambda i: np.dot(Xtest, weights[i].reshape(weights.shape[1],1) ), range(len(path))) )

    res_baseline={}
    res_baseline['weights'] = weights
    res_baseline['predictors'] = y_hat
    return res_baseline


def trainlasso1(Xtrain, Xtest, ytrain, path):
    weights = np.array(map(lambda x: train_lasso_3(Xtrain, ytrain, x), path))
    y_hat = np.array(map(lambda i: np.dot(Xtest, weights[i].reshape(weights.shape[1],1) ), range(len(path))) )

    res_baseline={}
    res_baseline['weights'] = weights
    res_baseline['predictors'] = y_hat
    return res_baseline


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


##### helper functions

def train_lasso_2(X,y,mu):
    model = linear_model.Lasso(alpha=mu, fit_intercept=False)
    model.fit(X * np.sqrt(X.shape[0]), y * np.sqrt(X.shape[0]))
    return model.coef_

def train_lasso_3(X,y,mu):
    model = linear_model.LassoLars(alpha=mu, fit_intercept=False)
    model.fit(X * np.sqrt(X.shape[0]), y * np.sqrt(X.shape[0]), Xy=None)
    return model.coef_

def train_lasso_4(X,y,mu):
    model = glmnet( x = X * np.sqrt(X.shape[0]), y = y * np.sqrt(X.shape[0]), family = 'gaussian', alpha = 1, nlambda = 20)
    w = glmnetCoef(model)
    #model.fit((X * np.sqrt(X.shape[0]), y * np.sqrt(X.shape[0]))
    #w = model.coef_
    return w


def load_toy_data(exms=100, feats=1000, non_zeros=100, sigma=0.1, corr=0.1, seed=None):
    # data generation, code taken from Adascreen implementation.
    if seed != None:
        np.random.seed(seed)
    # Generate data similar as done in the Sasvi paper
    X = np.random.uniform(low=0.,high=+1., size=(exms, feats))
    for i in range(1,feats):
        X[:,i] = (1.0-corr)*X[:,i] + corr*X[:,i-1]
    # (exms x features)
    beta_star = np.random.uniform(low=-1., high=+1., size=feats)
    cut = feats-non_zeros
    inds = np.random.permutation(range(feats))
    beta_star[inds[:cut]] = 0.0
    y = X.dot(beta_star) + sigma*np.random.rand(exms)
    return X, y, beta_star


    """
    paoki = NP.array(map(lambda x: train_lasso(SUX, SUy, x, rho, alpha, debug=debug)[0], path))
    if paoki.ndim != 1:
        paoki = paoki.reshape(paoki.shape[0], paoki.shape[1])
    lasso_obj(SUX, SUy, weights[15,:], path[15], weights[15,:]) - lasso_obj(SUX, SUy, beta[15,:], path[15], beta[15,:])
    list(map(lambda x: train_lasso(SUX,SUy,x,rho,alpha,debug=debug), path))
    w2, path2 = train_lasso_2(SUX, SUy)

    amountof_nz_inds = list(map(lambda x: len(x), nz_inds))
    amountof_screened_feat = list(map(lambda x: len(x), scr_inds))
    nz_inds_admm1 = list(map(lambda x: (x != 0).sum(), paok))


    #the objective value using ADMM is different from the monitor_lasso['objval'][-1]
    #incoorporate a test comparing the objective value, it has to be the same
    lasso_obj(SUX, SUy, np.reshape(paok.T[:, 15], (paok.T.shape[0], 1)), path[15], np.reshape(paok.T[:, 15], (paok.T.shape[0], 1))) - lasso_obj(SUX, SUy, np.reshape(beta[:, 15], (beta.shape[0], 1)), path[15], np.reshape(beta[:, 15], (beta.shape[0], 1)))
    objective_val_ADMM = lasso_obj(SUX, SUy, w, path[15], w) == lasso_obj(SUX, SUy, weights, path[15], weights)
    objective_val_ada = lasso_obj(SUX, SUy, w5, path[15], w5)
    ((SP.dot(SUX,beta[12,:])-SUy)**2).sum() - ((SP.dot(SUX,weights[12,:])-SUy)**2).sum()
    correctness_solvers= NP.array(map(lambda x: paok[x,0], range(len(path))))

    NP.sum((SUy - np.dot(SUX,weights[15,:].reshape(1000, 1)))**2) + path[15]*NP.sum(NP.abs(weights[15,:]))
    lasso_objective(SUy, SUX, weights[40,:], path[40]) - lasso_objective(SUy, SUX, beta[40,:], path[40])

     File "/home/aliki/Documents/hpi/llmlasso/LMM-Lasso/code/lmm_lasso.py", line 131, in train
    SUX = SUX * SP.tile(Sdi_sqrt,(n_f,1)).T
        MemoryError

    #idea: write a script that does the iteration for the lambda path and returns:
    #nz_inds_admm, time_admm, weights in an array
    start1 = time.time()
    paok = NP.array(map(lambda x: train_lasso(SUX,SUy,x,rho,alpha,debug=debug)[0:3], path))
    nzadmm = paok[:,2]
    time_admm = paok[:,1]
    #weightsadmm = paok[:,0]
    weights = NP.array(map(lambda x: paok[x,0], range(len(path))))
    if weights.ndim != 1:
        weights = weights.reshape(weights.shape[0], weights.shape[1])
    end1 = time.time()
    cythcode = end1 - start1
    
    #probably that part is never entered; why is there a second return statements in the end in one function
    # train null model
    S,U,ldelta0,monitor_nm = train_nullmodel(y,K,numintervals,ldeltamin,ldeltamax,debug=debug)
    
    # train lasso on residuals
    delta0 = SP.exp(ldelta0)
    Sdi = 1./(S+delta0)
    Sdi_sqrt = SP.sqrt(Sdi)
    SUX = SP.dot(U.T,X)
    SUX = SUX * SP.tile(Sdi_sqrt,(n_f,1)).T
    SUy = SP.dot(U.T,y)
    SUy = SUy * SP.reshape(Sdi_sqrt,(n_s,1))
    
    w,monitor_lasso = train_lasso(SUX,SUy,mu,rho,alpha,debug=debug)

    time_end = time.time()
    time_diff = time_end - time_start
    print '... finished in %.2fs'%(time_diff)

    res = {}
    res['ldelta0'] = ldelta0
    res['weights'] = w
    res['time'] = time_diff
    res['monitor_lasso'] = monitor_lasso
    res['monitor_nm'] = monitor_nm

    print 'this is never printed I guess'
    return res
    
    
    start2 = time.time()
    res_baseline = baseline_comparison.trainlasso(X[train_idx], X[test_idx], y[train_idx], lambda_path)
    end2 = time.time()
    baseline2 = end2 - start2
    
    corr_baseline = 1. / n_test * np.asarray([((res_baseline['predictors'][i]-res_baseline['predictors'][i].mean())*(y[test_idx]-y[test_idx].mean())).sum() / (
        res_baseline['predictors'][i].std() * y[test_idx].std())    for i in range(res_baseline['predictors'].shape[0])])
    ms_error_baseline = np.asarray([mse(res_baseline['predictors'][i], y[test_idx])   for i in range(yhat.shape[0])])
    nz_inds_baseline = list(map(lambda x: (x != 0).sum(), res_baseline['weights']))
       
        corr = [p[1] for p in correlations]
    corr_ada = [p[3] for p in correlations]
    corr_baseline = [p[5] for p in correlations]
    corr_tr = [p[0] for p in correlations]
    corr_ada_tr = [p[2] for p in correlations]
    corr_baseline_tr = [p[4] for p in correlations]
    ms_error = [p[1] for p in MSE]
    ms_error_ada = [p[3] for p in MSE]
    ms_error_baseline = [p[5] for p in MSE]
    ms_error_tr = [p[0] for p in correlations]
    ms_error_ada_tr = [p[2] for p in correlations]
    ms_error_baseline_tr = [p[4] for p in correlations]
    if inpycharm == True:
        pp = PdfPages(
                name.directory + '/paok/' + name.dataset + str(name.screening_rule) + name.phenotype + 'ranked SNPs' + '.pdf')
    else:
        pp = PdfPages(
                name.directory + '/moreplots/' + name.dataset + str(name.screening_rule) + name.phenotype + 'ranked SNPs' + '.pdf')
    #plt.plot(nrofSNPs, correlations, 'g^')
    plt.plot(nrofSNPs, corr_ada, 'bs', label="Adascreen solver")
    plt.plot(nrofSNPs, corr, 'rx', label=" ADMM solver")
    plt.plot(nrofSNPs, corr_baseline, 'g.', label="LASSO")
    plt.legend(loc='upper right', borderaxespad=0.)
    plt.axis([0, max(nrofSNPs) * 1.05, 0, 1.2])
    plt.title('phenotype:' + name.phenotype  +'\n' + 'correlations over nr of most important SNPs')
    plt.xlabel('#SNPs')
    pp.savefig()
    plt.close()

    #plt.plot(nrofSNPs, MSE, 'r^')
    plt.plot(nrofSNPs, ms_error_ada, 'bs', label="Adascreen solver")
    plt.plot(nrofSNPs, ms_error, 'rx', label=" ADMM solver")
    plt.plot(nrofSNPs, ms_error_baseline, 'g.', label="LASSO")
    plt.legend(loc='upper right', borderaxespad=0.)
    plt.axis([0, max(nrofSNPs) * 1.05, 0, max(max(ms_error, ms_error_ada, ms_error_baseline)) * 1.2])
    plt.title('MSE over nr of most important SNPs')
    plt.xlabel('#SNPs')
    pp.savefig()
    plt.close()
    pp.close()
    
    delta0 = SP.exp(ldelta0)
    Sdi = 1./(SP.sqrt(S)+delta0)         #== 1/(S+delta0), via SVD I have sqrt(singularval) = eigenval
    Sdi_sqrt = SP.sqrt(Sdi)
    SUX = SP.dot(U.T,X)
    SUX = SP.dot(SP.tile(Sdi_sqrt,(len(S),1)).T , SUX)
    X2 = X - SP.dot(U, SUX)
    SUX =np.vstack((SUX, X2))
    SUy = SP.dot(U.T,y)
    SUy = SUy * SP.reshape(Sdi_sqrt,(len(S),1))
    Y2 = y - SP.dot(U, SUy)
    SUy = np.vstack((SUy, Y2))
    """


def stability_selection(X, K, y, mu, n_reps, f_subset, **kwargs):
    """
    run stability selection

    Input:
    X: SNP matrix: n_s x n_f
    y: phenotype:  n_s x 1
    K: kinship matrix: n_s x n_s
    mu: l1-penalty

    n_reps:   number of repetitions
    f_subset: fraction of datasets that is used for creating one bootstrap

    output:
    selection frequency for all SNPs: n_f x 1
    """
    time_start = time.time()
    [n_s, n_f] = X.shape
    n_subsample = int(SP.ceil(f_subset * n_s))
    freq = SP.zeros(n_f)

    for i in range(n_reps):
        print 'Iteration %d' % i
        idx = SP.random.permutation(n_s)[:n_subsample]
        res = train(X[idx], K[idx][:, idx], y[idx], mu, **kwargs)
        snp_idx = (res['weights'] != 0).flatten()
        freq[snp_idx] += 1.

    freq /= n_reps
    time_end = time.time()
    time_diff = time_end - time_start
    print '... finished in %.2fs' % (time_diff)
    print 'the stability selection function'
    return freq


def train_lasso(X, y, mu, rho=1, alpha=1, max_iter=5000, abstol=1E-4, reltol=1E-2, zero_threshold=1E-3, debug=True):
    """
    train lasso via Alternating Direction Method of Multipliers:
    min_w  0.5*sum((y-Xw)**2) + mu*|z|

    Input:
    X: design matrix: n_s x n_f
    y: outcome:  n_s x 1
    mu: l1-penalty parameter
    rho: augmented Lagrangian parameter
    alpha: over-relatation parameter (typically ranges between 1.0 and 1.8)

    the implementation is a python version of Boyd's matlab implementation of ADMM-Lasso, which can be found at:
    http://www.stanford.edu/~boyd/papers/admm/lasso/lasso.html

    more information about ADMM can be found in the paper linked at:
    http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html

    In particular, you can use any other Lasso-Solver instead. For the experiments, reported in the paper,
    we used the l1-solver from the package scikits. We didn't apply it here to avoid third-party packages.
    """
    if debug:
        print '... train lasso'

    # init
    [n_s, n_f] = X.shape
    w = SP.zeros((n_f, 1))
    z = SP.zeros((n_f, 1))
    u = SP.zeros((n_f, 1))

    monitor = {}
    monitor['objval'] = []
    monitor['r_norm'] = []
    monitor['s_norm'] = []
    monitor['eps_pri'] = []
    monitor['eps_dual'] = []

    # calculate time
    startTime = time.time()

    # cache factorization
    U = factor(X, rho)

    # save a matrix-vector multiply
    Xy = SP.dot(X.T, y)

    if debug:
        print 'i\tobj\t\tr_norm\t\ts_norm\t\teps_pri\t\teps_dual'

    for i in range(max_iter):
        # w-update
        q = Xy + rho * (z - u)
        w = q / rho - SP.dot(X.T, LA.cho_solve((U, False), SP.dot(X, q))) / rho ** 2

        # z-update with relaxation
        zold = z
        w_hat = alpha * w + (1 - alpha) * zold
        z = soft_thresholding(w_hat + u, mu / rho)

        # u-update
        u = u + (w_hat - z)

        monitor['objval'].append(lasso_obj(X, y, w, mu, z))
        monitor['r_norm'].append(LA.norm(w - z))
        monitor['s_norm'].append(LA.norm(rho * (z - zold)))
        monitor['eps_pri'].append(SP.sqrt(n_f) * abstol + reltol * max(LA.norm(w), LA.norm(z)))
        monitor['eps_dual'].append(SP.sqrt(n_f) * abstol + reltol * LA.norm(rho * u))

        if debug:
            print '%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f' % (
            i, monitor['objval'][i], monitor['r_norm'][i], monitor['s_norm'][i], monitor['eps_pri'][i],
            monitor['eps_dual'][i])

        if monitor['r_norm'][i] < monitor['eps_pri'][i] and monitor['r_norm'][i] < monitor['eps_dual'][i]:
            break

    w[SP.absolute(w) < zero_threshold] = 0
    w = NP.asarray(w.flatten())
    totaltime = time.time() - startTime
    nonzeroadmm = (w != 0).sum()

    return w, totaltime, nonzeroadmm, monitor


def save_onlyeigenval(matirx):
    start1 = time.time()
    S = SP.linalg.eigvalsh(matirx, overwrite_a=True)
    end1 = time.time()
    scipycode = end1 - start1
    print 'the scipy.linalg.eigvalsh finished in %.5fs' % (scipycode)
    np.savetxt(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))) + "/moredata/foo_every25th.csv", S, delimiter=",")
    sys.exit("Error message - eigenvalue decomposition done")

def save_eigenval_eigenvec(matirx):
    #ccompute the eigenvalue decomposition and save it
    start1 = time.time()
    #U, S, Vh = SP.linalg.svd(matrix, full_matrices=True)
    eival, eivec = SP.linalg.eigh(matirx, overwrite_a=True)
    end1 = time.time()
    scipycode = end1 - start1
    print 'the scipy.linalg.eigh finished in %.5fs' % (scipycode)
    np.savetxt(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))) + "/moredata/eival_every50th.csv", eival, delimiter=",")
    np.savetxt(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))) + "/moredata/eivec_every50th.csv", eivec, delimiter=",")
    sys.exit("Error message - eigenvalue decomposition done")


def save_svd(matrix):
        #ccompute the SVD and save i((t
    print 'rows of matrix', matrix.shape[0]
    print 'rows of matrix', matrix.shape[1]
    start1 = time.time()
    _, S, V = SP.linalg.svd(matrix, lapack_driver='gesdd')
    end1 = time.time()
    scipycode = end1 - start1
    print 'the scipy.linalg.svd finished in %.5fs' % (scipycode)
    np.savetxt(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))) + "/moredata/svd_every40th_V.csv", V, delimiter=",")
    np.savetxt(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))) + "/moredata/svd_every40th_S.csv", S, delimiter=",")
    sys.exit("Error message - SVD done")

def sparse_eigenvec(matrix):
    # ccompute the eigenvalue decomposition and save it
    kernel = calculate_kernel(matrix)
    start1 = time.time()
    eival, eivec = SP.sparse.linalg.eigsh(kernel, which='LM', k=kernel.shape[0]-1)
    end1 = time.time()
    scipycode = end1 - start1
    print 'the scipy.sparse.linalg.eigsh finished in %.5fs' % (scipycode)
    #np.savetxt(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))) + "/moredata/eival_every50th.csv", eival, delimiter=",")
    #np.savetxt(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))) + "/moredata/eivec_every50th.csv", eivec, delimiter=",")
    sys.exit("Error message - eigenvalue decomposition done")

def calculate_kernel(X):
    thresh = float(0.05)
    n_f = X.shape[0]
    kernel = 1./n_f*SP.dot(X,X.T)
    kernel[np.logical_and(kernel < thresh, kernel > -thresh)] = 0
    return kernel

def prioritized_SNps(X, y):
    pvalues = NP.array(map(lambda j: SP.stats.linregress(X[:,j], y.flatten())[3], range(X.shape[1]) ))
    ordered_indices = NP.argsort(pvalues)
    #important_X = X[:, ordered_indices[:numberofSNPs]]
    #important_X = SNP_standard(important_X)     #otherwise nullmodel results in complex numbers
    return ordered_indices

def factor(X,rho):
    """
    computes cholesky factorization of the kernel K = 1/rho*XX^T + I

    Input:
    X design matrix: n_s x n_f (we assume n_s << n_f)
    rho: regularizaer

    Output:
    L  lower triangular matrix
    U  upper triangular matrix
    """
    n_s,n_f = X.shape
    K = 1/rho*SP.dot(X,X.T) + SP.eye(n_s)
    #try numpy cholesky decomp, where NP.allclose(LA.cholesky(K), U.T) so return U.T instead of U
    U = NP.linalg.cholesky(K)
    #U = LA.cholesky(K)

    print 'the factor function'
    return U.T


def soft_thresholding(w,kappa):
    """
    Performs elementwise soft thresholding for each entry w_i of the vector w:
    s_i= argmin_{s_i}  rho*abs(s_i) + rho/2*(x_i-s_i) **2
    by using subdifferential calculus

    Input:
    w vector nx1
    kappa regularizer

    Output:
    s vector nx1
    """
    n_f = w.shape[0]
    zeros = SP.zeros((n_f,1))
    s = NP.max(SP.hstack((w-kappa,zeros)),axis=1) - NP.max(SP.hstack((-w-kappa,zeros)),axis=1)
    s = SP.reshape(s,(n_f,1))
    return s

