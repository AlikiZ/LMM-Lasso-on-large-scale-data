#from pandas_plink import read_plink
import os
from pyplink import PyPlink
from tqdm import tqdm
import numpy as np



def read_genomic_data(chromosome_path, samples):
    #filename = os.path.join(BASE_GEN, prefix)
    bed = PyPlink(chromosome_path)
    fam = bed.get_fam()
    # bim = bed.get_bim()

    id_to_index = dict((val, i) for i, val in enumerate(fam.iid.values))

    reduced_genotypes = []
    all_markers = []
    # for (marker_id, genotypes), i in tqdm(zip(bed,range(1000))):
    for marker_id, genotypes in tqdm(bed):
        reduced_genotypes.append(genotypes[:samples])
        all_markers.append(marker_id)
    reduced_genotypes = np.array(reduced_genotypes).T
    return reduced_genotypes

#path = ['ukb_chr' + str(chromosome) + '_v2' for chromosome in range(21, 22)]
#chromosome_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))) + '/moredata/data_ukb/' + str(path[0])
#path = '/path/to/data'
#X = read_genomic_data(chromosome_path, 10000)
#print X.shape

#path = '/path/to/data'
#G = read_plink1_bin(chr_path+ ".bed", verbose=True)
#print(G.values)
#print(G)

#(bim, fam, bed) = read_plink(join(get_data_folder(), "data"), verbose=False)
#print(bim.head())
#(bim, fam, G) = read_plink(chr_path)
#print(G.values)
