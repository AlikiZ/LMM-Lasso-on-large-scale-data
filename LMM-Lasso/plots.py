import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
import os
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def along_lambdapath(lambda_path, corr_ada, lmax, ms_error_ada, path):
    # , nz_inds, nz_inds_admm
    pp = PdfPages(path + '/moreplots/' + name.dataset + str(name.screening_rule) + name.phenotype + '.pdf')
    # plot path vs correlation
    plt.plot(lambda_path, corr_ada, 'bs', label="with screening")
    plt.legend(loc='upper right', borderaxespad=0.)
    plt.axis([0, max(lambda_path) * 1.2, 0, 1.2])
    plt.vlines(lmax, 0, 1.2, colors='k', linestyles='dashed')
    plt.title('Pearson correlation')
    plt.xlabel('lambda path')
    plt.ylabel('correlation')
    pp.savefig()
    plt.close()

    # plot path vs mse
    plt.plot(lambda_path, ms_error_ada, 'bs', label="with screening")
    plt.legend(loc='upper right', borderaxespad=0.)
    plt.axis([0, max(lambda_path) * 1.2, 0, max(ms_error_ada) * 1.1])
    plt.vlines(lmax, 0, max(ms_error_ada) * 1.1, colors='k', linestyles='dashed')
    plt.title('mean squared error ')
    plt.xlabel('lambda path')
    plt.ylabel('MSE')
    pp.savefig()
    plt.close()
    pp.close()


def along_lambdapath_compare(inpycharm, name, lambda_path, corr_ada, corr, corr_baseline, lmax, ms_error, ms_error_ada, ms_error_baseline, nz_inds, nz_inds_admm, nz_inds_baseline, number_screened_feat):
    # y_test_idx, yhat_maxidx, maxidx, y_ada, baseline_predictors_maxidx,
    if inpycharm == True:
        pp = PdfPages(
            name.directory + '/paok/' + name.dataset + str(name.screening_rule) + name.phenotype + '.pdf')
    else:
        pp = PdfPages(
            name.directory + '/moreplots/' + name.dataset + str(name.screening_rule) + name.phenotype + '.pdf')
    """
    # plt.figure(2, figsize=(9,3))
    plt.plot(y_test_idx, yhat_maxidx, 'bx', y_test_idx, y_test_idx, 'k')
    plt.xlabel('y(true)')
    plt.ylabel('y(predicted)')
    plt.title(
        'phenotype:' + name.phenotype + '\n' + 'test data (ADMM) for the best lambda {0:.2f}'.format(
            lambda_path[maxidx]))
    pp.savefig()
    plt.close()

    hist, bins = np.histogram(y_test_idx.reshape(y_ada.shape[1], y_ada.shape[2]) - y_ada[maxidx], bins=20)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.title('histogram y - y(predicted) with Adascreen')
    pp.savefig()
    plt.close()

    plt.plot(y_test_idx, y_ada[maxidx], 'bx', y_test_idx, y_test_idx, 'k')
    plt.xlabel('y(true)')
    plt.ylabel('y(predicted)')
    plt.title('test data (Ada) for the best lambda {0:.2f}'.format(lambda_path[maxidx]))
    pp.savefig()
    plt.close()
    
    plt.plot(y_test_idx, baseline_predictors_maxidx, 'bx', y_test_idx, y_test_idx, 'k')
    plt.xlabel('y(true)')
    plt.ylabel('y(predicted)')
    plt.title('test data (baseline solver) for the best lambda {0:.2f}'.format(lambda_path[maxidx]))
    pp.savefig()
    plt.close()
    """
    # plot path vs correlation
    plt.plot(lambda_path, corr_ada, 'bs', label="Adascreen solver")
    plt.plot(lambda_path, corr, 'rx', label=" ADMM solver")
    plt.plot(lambda_path, corr_baseline, 'g.', label="LASSO")
    plt.legend(loc='upper right', borderaxespad=0.)
    plt.axis([0, max(lambda_path) * 1.2, 0, 1.2])
    plt.vlines(lmax, 0, 1.2, colors='k', linestyles='dashed')
    plt.title('correlation y - y(predicted)')
    plt.xlabel('lambda path')
    plt.ylabel('correlation')
    pp.savefig()
    plt.close()

    # plot path vs mse
    plt.plot(lambda_path, ms_error_ada, 'bs', label="Adascreen solver")
    plt.plot(lambda_path, ms_error, 'rx', label=" ADMM solver")
    plt.plot(lambda_path, ms_error_baseline, 'g.', label="LASSO")
    plt.legend(loc='upper right', borderaxespad=0.)
    plt.axis([0, max(lambda_path) * 1.2, 0, max(ms_error) * 1.1])
    plt.vlines(lmax, 0, max(ms_error) * 1.1, colors='k', linestyles='dashed')
    plt.title('mean squared error ')
    plt.xlabel('lambda path')
    plt.ylabel('MSE')
    pp.savefig()
    plt.close()

    # plot path vs amount of non-zero features
    plt.plot(lambda_path, nz_inds, 'bs', label="Adascreen solver")
    plt.plot(lambda_path, nz_inds_admm, 'rx', label=" ADMM solver")
    plt.plot(lambda_path, nz_inds_baseline, 'g.', label="LASSO")
    plt.legend(loc='upper right', borderaxespad=0.)
    plt.axis([0, max(lambda_path) * 1.2, 0, max(nz_inds) + 10])
    plt.vlines(lmax, 0, max(nz_inds) + 10, colors='k', linestyles='dashed')
    plt.title('amount of non-zero features ')
    plt.xlabel('lambda path')
    plt.ylabel('#non-zero features')
    pp.savefig()
    plt.close()

    # plot path vs amount of screened features
    plt.plot(lambda_path, number_screened_feat, 'g^')
    plt.axis([0, max(lambda_path) * 1.2, 0, max(number_screened_feat) + 10])
    plt.vlines(lmax, 0, max(number_screened_feat) + 10, colors='k', linestyles='dashed')
    plt.title('amount of screened features with Adascreen')
    plt.xlabel('lambda path')
    plt.ylabel('#screened features')
    pp.savefig()
    plt.close()
    pp.close()


def alongSNPnr(inpycharm, correlations, MSE, nrofSNPs, name):
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
    a = np.reshape(corr_tr + corr_ada_tr + corr_baseline_tr, (3, len(corr_tr)))
    a.astype(int)
    b = np.reshape(ms_error_tr + ms_error_ada_tr + ms_error_baseline_tr, (3, len(ms_error_tr)))
    b.astype(int)

    x = nrofSNPs
    y = ['ADMM', 'LMM-LASSO', 'LASSO']

    if inpycharm == True:
        pp = PdfPages(
                name.directory + '/paok/' + name.dataset + str(name.screening_rule) + name.phenotype + 'ranked SNPs' + '.pdf')
    else:
        pp = PdfPages(
                name.directory + '/moreplots/' + name.dataset + str(name.screening_rule) + name.phenotype + 'ranked SNPs' + '.pdf')

    plt.plot(nrofSNPs, corr_ada, 'bs', label="Adascreen solver")
    plt.plot(nrofSNPs, corr, 'rx', label=" ADMM solver")
    plt.plot(nrofSNPs, corr_baseline, 'g.', label="LASSO")
    plt.legend(loc='upper right', borderaxespad=0.)
    plt.axis([0, max(nrofSNPs) * 1.05, 0, 1.2])
    plt.title('phenotype:' + name.phenotype + '\n' + 'correlations over nr of most important SNPs')
    plt.xlabel('#SNPs')
    pp.savefig()
    plt.close()

    # plt.plot(nrofSNPs, MSE, 'r^')
    plt.plot(nrofSNPs, ms_error_ada, 'bs', label="Adascreen solver")
    plt.plot(nrofSNPs, ms_error, 'rx', label=" ADMM solver")
    plt.plot(nrofSNPs, ms_error_baseline, 'g.', label="LASSO")
    plt.legend(loc='upper right', borderaxespad=0.)
    plt.axis([0, max(nrofSNPs) * 1.05, 0, max(max(ms_error, ms_error_ada, ms_error_baseline)) * 1.2])
    plt.title('MSE over nr of most important SNPs')
    plt.xlabel('#SNPs')
    pp.savefig()
    plt.close()


    fig = plt.figure(3)
    ax1 = plt.subplot(2, 1, 1)
    cmap = matplotlib.colors.ListedColormap(['orange', 'green'])
    boundaries = [0, 0.5, 1]
    heatplot1 = ax1.imshow(a, cmap=cmap)
    ax1.set_xticklabels(['1'] + x)
    ax1.set_yticklabels([0] + y)
    tick_spacing = 1
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax1.set_xticks(np.arange(a.shape[1] + 1) - .5, minor=True)
    ax1.set_yticks(np.arange(a.shape[0] + 1) - .5, minor=True)
    ax1.grid(which="minor", color="w", linestyle='-', linewidth=15)
    ax1.tick_params(which="minor", bottom=False, left=False)
    ax1.set_title("All the weights = 0")
    # ax1.set_xlabel('Number of SNPs')
    ax1.set_ylabel('correlation')
    cbar = plt.colorbar(heatplot1, cmap=cmap, boundaries=boundaries, ticks=[0, 1], shrink=0.7)
    cbar.ax.set_yticklabels(['False', 'True'])
    #cbar.set_label('boolean values', rotation=270)

    ax2 = plt.subplot(2, 1, 2)
    heatplot2 = ax2.imshow(b, cmap=cmap)
    ax2.set_xticklabels(['1'] + x)
    ax2.set_yticklabels([0] + y)
    tick_spacing = 1
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax2.set_xticks(np.arange(a.shape[1] + 1) - .5, minor=True)
    ax2.set_yticks(np.arange(a.shape[0] + 1) - .5, minor=True)
    ax2.grid(which="minor", color="w", linestyle='-', linewidth=15)
    ax2.tick_params(which="minor", bottom=False, left=False)
    # ax2.set_title("All the weights are equal to zero MSE")
    ax2.set_xlabel('Number of SNPs')
    ax2.set_ylabel('MSE')
    cbar2 = plt.colorbar(heatplot2, cmap=cmap, boundaries=boundaries, ticks=[0, 1], shrink=0.7)
    cbar2.ax.set_yticklabels(['False', 'True'])
    #cbar2.set_label('boolean values', rotation=270)
    pp.savefig()
    plt.close()
    pp.close()


def plot_speedup(X, res_time , name, lmax, solver, inpycharm, screening_rules=None, path= None, times_solver= None):
    """
    props = ExperimentViewProperties('Speed-up Comparison', '$\lambda / \lambda_{max}$', 'Speed-up', loc=1,
                                     xscale='log')
    props.setStats(X)
    props.names.append('Path solver w/o screening')
    """
    screening_rules = [screening_rules]
    res = np.ones((len(screening_rules)+1, len(path)))

    for s in range(len(screening_rules)):
        #props.names.append(screening_rules[s])
        times = np.asarray(times_solver)
        for i in range(1, len(path)):
            res[s, i] = float(np.sum(times[:i]))

    #res_time = np.array(map(lambda x: train_lasso(X, y, x, rho=1, alpha=1, debug=False)[1], path))
    res[-1, :] = res_time

    if inpycharm == True:
        pp = PdfPages(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))) + '/paok/' + name.dataset + name.phenotype + 'time_comparison.pdf')
    else:
        pp = PdfPages(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))) + '/moreplots/' + name.dataset + name.phenotype + 'time_comparison.pdf')

    plt.figure(1, figsize=(9, 3))
    plt.plot(path, res[0, :], 'm.', label="Adascreen solver (" + str(solver) + ")")
    plt.plot(path, res[1, :], 'gx', label="ADMM solver")
    plt.legend(loc='upper right', borderaxespad=0.)
    plt.axis([0, max(path) * 1.2, 0, np.amax(res)])
    plt.title('time comparison')
    plt.xlabel('lambda path')
    plt.ylabel('time (in sec)')
    plt.vlines(lmax, 0, np.amax(res), colors='k', linestyles='dashed')
    pp.savefig()
    #plt.savefig( os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))) + '/moreplots/' + str(screening_rules[0]) + str(dataset) + 'time_comparison.pdf')
    plt.close()

    #res is now the ratio
    for i in range(1, len(path)):
        res[:, i] = res_time[i] / res[:, i]

    plt.figure(1, figsize=(9, 3))
    plt.plot(path, res[0, :], 'm.', label="Adascreen solver (" + str(solver) + ")")
    plt.plot(path, res[1, :], 'gx', label="ADMM solver")
    plt.legend(loc='upper right', borderaxespad=0.)
    plt.axis([0, max(path) * 1.2, 0, np.amax(res)])
    plt.title('speed up')
    plt.xlabel('lambda path')
    plt.ylabel('Adascreen (x) times faster')
    plt.vlines(lmax, 0, np.amax(res), colors='k', linestyles='dashed')
    pp.savefig()
    #plt.savefig(
    #   os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))) + '/moreplots/' + str(screening_rules[0]) + str(
    #        dataset) + 'experiment' + 'time_comparison.pdf')
    plt.close()
    pp.close()

    #props.plot(x, res, np.zeros(res.shape), save_pdf=True, directory='/home/aliki/Documents/hpi/paok/fun/')
    #plt.close()
    #plt.close('all')
    #x = np.zeros(len(path))
    #x[0] = 1.0
    #x[i] = path[i] / path[0]
