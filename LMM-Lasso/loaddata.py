import pandas as pd
from pysnptools.snpreader import Bed
import os
import numpy as np
from namesforplotting import naming
from lmm_lasso import SNP_standard
from read_genetics import read_genomic_data
import time
from lmm_lasso import leastFrequent


def loadSnps(samples, everynthfeature, path = '/home/aliki/uni/moredata/data_ukb/' + 'ukb_chr22_v2.bed'):
    #it has to share the same name in all formats .bed , .bim , .fam
    snps =Bed(path, count_A1=False)
    if type(samples) == int and everynthfeature != 1:
        snps = snps[:samples,:: everynthfeature]
    if everynthfeature == 1:
        snps = snps[:samples,:]
    #read every n-th so as to have less then everynthfeature
    if samples == 'all':
        snps = snps[:, :: everynthfeature]
    snp_data = snps.read()
    #can be snp_df = pd.DataFrame(data=snp_data.val, index=snp_data._row[:, 0],columns=snp_data._col), w/o dtype=int8
    snp_df = pd.DataFrame(data=snp_data.val, index=snp_data._row[:, 0],columns=snp_data._col, dtype='int8')
    #in case there are na values drop the whole axis:  snp_df = snp_df.dropna(axis=1)
    return snp_df


def loadukb_server(samples, everynthfeature, chromosome_path = 'ukb_chr22_v2.bed'):
    chromosome_path = ['/mnt/30T/data/ukbiobank/original/genetics/microarray/unzipped/' + file for file in chromosome_path]
    csvdata = pd.read_csv('/mnt/30T/data/ukbiobank/derived/projects/LMM-Lasso' + '/clean_standingheight.csv', index_col=0)
    csvdata.columns = [ 'standingheight']
    starttime = time.time()
    # loading with pysnptools
    Xi = loadSnps(samples, everynthfeature, chromosome_path[0]+'.bed')
    print chromosome_path[0]
    for path in chromosome_path[1:]:
        Xi = pd.concat([Xi, loadSnps(samples, everynthfeature, path+'.bed')], axis=1, copy=False)
        print path, Xi.shape 
    # make the index a numeric value
    Xi.set_index(pd.Series(Xi.index.values.astype(int)), inplace=True)
    endload = time.time()
    print 'time of loading with pysnptools', endload - starttime
    # filter for MAF - not needed in case a QC was already performed
    #marker_MAF3 = np.asarray(pd.read_csv("/mnt/30T/aliki" + "/marker_MAF3.csv", header=None, dtype = int, squeeze=True))
    """
    marker_MAF3, marker_MAF1 = filterMAF(Xi)
    Xi = Xi.drop(columns=Xi.columns[marker_MAF3])
    """
    # make sure to use the genotype-phenotype info for the same person
    commonindeces = list(set(Xi.index.values).intersection(set(csvdata.index.values)))
    Xi = Xi.iloc[Xi.index.isin(commonindeces), :].sort_index()
    csvdata = csvdata.iloc[csvdata.index.isin(commonindeces), :].sort_index()
    print set(Xi.index.values == csvdata.index.values)
    # standardize the SNP values column-wise
    Xi = SNP_standard(Xi.values)
    print 'time of transforming design matrix X', time.time() - endload
    name = naming()
    phenotype = 'standing_height'
    name.phenotype(phenotype)
    y = np.asarray(csvdata['standingheight'])
    print 'total time (including reading) of getting the data ready', time.time() - starttime
    return Xi, y, name


"""  helper functions, none of the following is used when a QC is already done   """

def loadukb_local(samples, features, chromosome_path = 'ukb_chr22_v2.bed'):
    #here the IDs of X and y are not overlapping
    csvdata = pd.read_csv(os.path.abspath(
        os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))) + '/uni/moredata/data_ukb/clean_standingheight.csv', index_col=0)
    csvdata.columns = [ 'standingheight']
    chromosome_path = [
        os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))) + '/uni/moredata/data_ukb/' + file
        for file in chromosome_path]
    """     that is another way to read in the data
    Xi = read_genomic_data(chromosome_path[0], samples)
    print chromosome_path[0], Xi.shape
    for path in chromosome_path[1:]:
        Xi = np.append(Xi, read_genomic_data(path, samples), axis=1)
        print path, Xi.shape
    """
    Xi = loadSnps(samples, features, chromosome_path[0])
    print chromosome_path[0]
    for path in chromosome_path[1:]:
        # test if the chromosomes share the same people;s structure
        if not bool(set( loadSnps(samples, features, path).index.values == Xi.index.values )):
            print 'in the above path the IDs of the ppl are mixed up'
        Xi = pd.concat([Xi, loadSnps(samples, features, path)], axis=1, copy=False)
        print path, Xi.shape
    # make the index a numeric value
    Xi.set_index(pd.Series(Xi.index.values.astype(int)), inplace=True)
    # find and exclude SNPs with MAF < 3%
    marker_MAF3 = []
    for i in range(Xi.shape[1]):
        if float(leastFrequent(np.asarray(Xi.iloc[:, i]))) / float(Xi.shape[1]) < 0.03:
            marker_MAF3.append(i)
    Xi = Xi.drop(columns=Xi.columns[marker_MAF3])
    commonindeces = list(set(Xi.index.values).intersection(set(csvdata.index.values)))
    Xi = Xi.iloc[Xi.index.isin(commonindeces), :].sort_index()
    csvdata = csvdata.iloc[csvdata.index.isin(commonindeces), :].sort_index()
    print 'Are genotype and phenotype corresponding to the same person?', set(Xi.index.values == csvdata.index.values)
    Xi = SNP_standard(Xi.values)
    name = naming()
    phenotype = 'standing_height'
    name.phenotype(phenotype)
    y = np.asarray(csvdata['standingheight'])
    return Xi, y, name


def Frequency(arr):
    n = len(arr)
    # Insert all elements in Hash.
    Hash = dict()
    for i in range(n):
        if arr[i] in Hash.keys():
            Hash[arr[i]] += 1
        else:
            Hash[arr[i]] = 1
    #define the counts: h:= homozygous (major allele) 0, het := heterozygous 1, ma:=minor allele 2, mv:= missing values -1
    h = np.inf
    het = np.inf
    ma = np.inf
    missing_v = 0
    for i in range(len(Hash.keys())):
        if Hash.keys()[i] == 0:
            h = Hash.values()[i]
        if Hash.keys()[i] == 1:
            het = Hash.values()[i]
        if Hash.keys()[i] == -1:
            missing_v = Hash.values()[i]
        else:
            ma = Hash.values()[i]

    # find the min absolute frequency,only among the SNPs 0,1,2
    min_count = min(h, het, ma) #min(Hash.values())
    #find the most frequent value, even if the missing value is the most frequent one it will be deleted
    #because of the seleccted threshold if missing_v > 0.1  so another value will be the most frequent
    most_freq = max(Hash.iterkeys(), key=Hash.get)

    return min_count, missing_v, most_freq


#exclude a column when MAF3, missing value over 10%, and impute with the most frequent value
def exclude(arr):
    min_count, missing_v, most_freq = Frequency(arr)
    bool_marker = True

    if float(min_count) / float(len(arr)) < 0.03:
        bool_marker = False
    if float(missing_v) / float(len(arr)) > 0.1:
        bool_marker = False
    else:
        arr = np.where(arr == -1, most_freq, arr)

    return bool_marker, arr

def filterMAF(Xi):
    # filter MAF, Xi is pandas dataframe
    marker_MAF3 = []
    marker_MAF1 = []
    for i in range(Xi.shape[1]):
        if float(leastFrequent(np.asarray(Xi.iloc[:, i]))) / float(Xi.shape[1]) < 0.03:
            marker_MAF3.append(i)
            if float(leastFrequent(np.asarray(Xi.iloc[:, i]))) / float(Xi.shape[1]) < 0.01:
                marker_MAF1.append(i)
    return marker_MAF3, marker_MAF1