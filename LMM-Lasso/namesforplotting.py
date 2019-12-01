

class naming(object):

    def __init__(self):
        self.dataset = ""

    def features(self, dataset, screening_rule, directory):
        self.dataset = dataset
        self.screening_rule = screening_rule
        self.directory = directory

    def phenotype(self, phenotype):
        self.phenotype = phenotype

    def solver(self, solver):
        self.solver = solver

    def dataset(self, dataset):
        self.dataset = str(dataset)

#name = naming()


#functions used as a scratch version
def train_lasso_2(X, y, geomul=0.9, lower_bound=0.001, steps=65):
    # import less stuff to only find weights and ??
    solver = [SklearnCDSolver(), SklearnLarsSolver(),
              ProximalGradientSolver(), AccelProximalGradientSolver()]  # ActiveSetCDSolver, GlmnetSolver

    myLasso = ScreeningLassoPath(DOME(), solver[1], path_lb=lower_bound, path_steps=steps, path_stepsize=geomul,
                                 path_scale='geometric')
    beta, nz_inds, scr_inds, path, times_solver, times_screening = myLasso.fit(X.T, y, max_iter=1000, tol=1e-4,
                                                                               debug=False)
    weights = beta[:, 15]
    weights = NP.reshape(weights, (X.shape[1], 1))
    timescreenandsolve = times_solver[15] + times_screening[15]
    return weights, path

def train_lasso_3(X, y, geomul=0.9, lower_bound=0.001, steps=65):
    # import less stuff to only find weights and ??
    solver = [SklearnCDSolver(), SklearnLarsSolver(),
              ProximalGradientSolver(), AccelProximalGradientSolver()]  # ActiveSetCDSolver, GlmnetSolver

    myLasso = ScreeningLassoPath(StrongRule(), solver[1], path_lb=lower_bound, path_steps=steps, path_stepsize=geomul,
                                 path_scale='geometric')
    beta, nz_inds, scr_inds, path, times_solver, times_screening = myLasso.fit(X.T, y, max_iter=1000, tol=1e-4,
                                                                               debug=False)
    weights = beta[:, 15]
    weights = NP.reshape(weights, (X.shape[1], 1))
    timescreenandsolve = times_solver[15] + times_screening[15]
    return weights, path

def train_lasso_4(X, y, geomul=0.9, lower_bound=0.001, steps=65):
    # import less stuff to only find weights and ??
    solver = [SklearnCDSolver(), SklearnLarsSolver(),
              ProximalGradientSolver(), AccelProximalGradientSolver()]  # ActiveSetCDSolver, GlmnetSolver

    myLasso = ScreeningLassoPath(SAFE(), solver[1], path_lb=lower_bound, path_steps=steps, path_stepsize=geomul,
                                 path_scale='geometric')
    beta, nz_inds, scr_inds, path, times_solver, times_screening = myLasso.fit(X.T, y, max_iter=1000, tol=1e-4,
                                                                               debug=False)
    weights = beta[:, 15]
    weights = NP.reshape(weights, (X.shape[1], 1))
    timescreenandsolve = times_solver[15] + times_screening[15]
    return weights, path

def train_lasso_6(X, y, geomul=0.9, lower_bound=0.001, steps=65):
    #import less stuff to only find weights and ??
    solver = [SklearnCDSolver(), SklearnLarsSolver(),
              ProximalGradientSolver(), AccelProximalGradientSolver()]   #ActiveSetCDSolver, GlmnetSolver

    myLasso = ScreeningLassoPath(EDPP(), solver[1], path_lb=lower_bound, path_steps=steps, path_stepsize=geomul,
                                path_scale='geometric')
    beta, nz_inds, scr_inds, path, times_solver, times_screening = myLasso.fit(X.T, y, max_iter=1000, tol=1e-4, debug=False)
    weights = beta[:, 15]
    weights = NP.reshape(weights, (X.shape[1], 1))
    timescreenandsolve = times_solver[15] + times_screening[15]
    return weights, path


#sftp://aliki@172.20.24.26/mnt/30T/data/ukbiobank/original/genetics/microarray/EGAD00010001497/ukb_cal_chr22_v2.bed.gz
#sftp://aliki@172.20.24.26/mnt/30T/data/ukbiobank/original/genetics/microarray/EGAD00010001497/ukb_snp_chr22_v2.bim.gz

    # after pip install cprofilev
    # python -m cProfile -o output.profile test.py
    # cprofilev -f output.profile

    # visualize with KCacheGrind,firstly: source activate py2
    # pyprof2calltree -i prof.out -k     pyprof2calltree -i output.profileallSNPs -k

    # ~/Documents/master-thesis/LMM-Lasso/code in output.profile, output.profileallSNPs

#loading different data sets: 1. Aradopsis Thaliana 2. synthetic data
    """
        # load genotypes
        geno_filename = os.path.join(data_dir,'genotypes.csv')
        X = SP.genfromtxt(geno_filename)
        [n_s,n_f] = X.shape

        # simulate phenotype
        SP.random.seed(1)
        n_c = 5
        idx = SP.random.randint(0,n_f,n_c)
        w = 1./n_c * SP.ones((n_c,1))
        ypheno = SP.dot(X[:,idx],w)
        ypheno = (ypheno-ypheno.mean())/ypheno.std()
        pheno_filename = os.path.join(data_dir,'poppheno.csv')
        ypop = SP.genfromtxt(pheno_filename)
        ypop = SP.reshape(ypop,(n_s,1))
        y = 0.3*ypop + 0.5*ypheno + 0.2*SP.random.randn(n_s,1)
        y = (y-y.mean())/y.std()

        # init
        debug = False
        n_train = 150
        n_test = n_s - n_train
        n_reps = 100
        f_subset = 0.5
        mu = 10
        dataset = "semi-empirical"
        """

    """
            #synthetic data that is correlated
            X, y, _ = lmm_lasso.load_toy_data()
            y = (y-y.mean())/y.std()
            # init
            [n_s,n_f] = X.shape
            debug = False
            n_train = int(n_s * 0.7)
            n_test = n_s - n_train
            n_reps = 100
            f_subset = 0.5
            mu = 10
            dataset = "synthetic"
            """




    """
    corrv1 = 1./n_test * np.asarray([((yhat[i]-yhat[i].mean())*(y[test_idx]-y[test_idx].mean())).sum() / (
                    yhat[i].std()*y[test_idx].std())       for i in range(y_ada.shape[0])])
    corr_adav1 = 1. / n_test * np.asarray([((y_ada[i]-y_ada[i].mean())*(y[test_idx]-y[test_idx].mean())).sum() / (
                    y_ada[i].std() * y[test_idx].std())    for i in range(y_ada.shape[0])])
    corr_baselinev1 = 1. / n_test * np.asarray([((res_baseline['predictors'][i] - res_baseline['predictors'][
            i].mean()) * (y[test_idx] - y[test_idx].mean())).sum() / (res_baseline['predictors'][i].std() * y[test_idx].std())   for i in range(res_baseline['predictors'].shape[0])])
            
    # stability selection
    ss = lmm_lasso.stability_selection(X,K,y,mu,n_reps,f_subset)

    # create plot folder
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # plot kernel
    fig = plt.figure()
    fig.add_subplot(111)
    plt.imshow(K,interpolation='nearest')
    plt.xlabel('samples')
    plt.ylabel('samples')
    plt.title('Population Kernel')
    fn_out = os.path.join(plots_dir,'kernel.pdf')
    plt.savefig(fn_out)
    plt.close()

    # plot negative log likelihood of the null model
    monitor = res['monitor_nm']
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(monitor['ldeltagrid'],monitor['nllgrid'],'b-')
    plt.plot(monitor['ldeltaopt'],monitor['nllopt'],'r*')
    plt.xlabel('ldelta')
    plt.ylabel('negative log likelihood')
    plt.title('nLL on the null model')
    fn_out = os.path.join(plots_dir, 'nLL.pdf')
    plt.savefig(fn_out)
    plt.close()

    # plot Lasso convergence
    monitor = res['monitor_lasso']
    fig = plt.figure()
    fig.add_subplot(311)
    plt.plot(monitor['objval'])
    plt.title('Lasso convergence')
    plt.ylabel('objective')
    fig.add_subplot(312)
    plt.plot(monitor['r_norm'],'b-',label='r norm')
    plt.plot(monitor['eps_pri'],'k--',label='eps pri')
    plt.ylabel('r norm')
    fig.add_subplot(313)
    plt.plot(monitor['s_norm'],'b-',label='s norm')
    plt.plot(monitor['eps_dual'],'k--',label='eps dual')
    plt.ylabel('s norm')
    plt.xlabel('iteration')
    fn_out = os.path.join(plots_dir,'lasso_convergence.pdf')
    plt.savefig(fn_out)
    plt.close()

    # plot weights
    fig = plt.figure()
    fig.add_subplot(111)
    plt.title('Weight vector')
    plt.plot(w,'b',alpha=0.7)
    for i in range(idx.shape[0]):
        plt.axvline(idx[i],linestyle='--',color='k')
    fn_out = os.path.join(plots_dir,'weights.pdf')
    plt.savefig(fn_out)
    plt.close()

    # plot stability selection
    fig = plt.figure()
    fig.add_subplot(111)
    plt.title('Stability Selection')
    plt.plot(ss,'b',alpha=0.7)
    for i in range(idx.shape[0]):
        plt.axvline(idx[i],linestyle='--',color='k')
    plt.axhline(0.5,color='r')
    fn_out = os.path.join(plots_dir,'ss_frequency.pdf')
    plt.savefig(fn_out)
    plt.close()

    # plot predictions
    fig = plt.figure()
    fig.add_subplot(111)
    plt.title('prediction')
    plt.plot(y[test_idx],yhat, 'bx')
    plt.plot(y[test_idx],y[test_idx],'k')
    plt.xlabel('y(true)')
    plt.ylabel('y(predicted)')
    plt.xlabel('SNPs')
    plt.ylabel('weights')
    fn_out = os.path.join(plots_dir,'predictions.pdf')
    plt.savefig(fn_out)
    plt.close()
    
    import statsmodels.api as sm
    model = sm.OLS(y.flatten(), X).fit()
    predictions = model.predict(X)
    print_model = model.summary()
    
    
                    coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1             0.0043        inf          0        nan         nan         nan
x2            -0.0033        inf         -0        nan         nan         nan
x3            -0.0036        inf         -0        nan         nan         nan
x4            -0.0002        inf         -0        nan         nan         nan

map(lambda j: len(np.unique(a[:,j])), range(a.shape[1]) )
np.allclose(a, a1)

    start1 = time.time()
    x1 = example.prioritized_SNps(X, y, numberofSNPs)
    x2 = example.prioritized_SNpsv2(X, y, numberofSNPs)
    end1 = time.time()
    cythcode = end1 - start1
    
    """




