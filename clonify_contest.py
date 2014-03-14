from __future__ import print_function
import sys
import time
import math
import json
import numpy as np
from multiprocessing import Pool, cpu_count
import fastcluster as fc
from Bio import pairwise2
from Levenshtein import distance
from scipy.cluster.hierarchy import fcluster


default_dtype = 'f4'
distance_cutoff = 0.32


class Seq(object):
    """Contains genetic characteristics for a single sequence.
       Input:
       data = a MongoDB result (dict-like) containing the following fields:
                [seq_id, v_gene, j_gene, <junc_query>, var_muts_nt]
                where <junc_query> is the sequence of the nucleotide or AA junction.
       junc_query = either 'junc_aa' or 'junc_nt' for nucleotide or AA junctions, respectively.
    """
    def __init__(self, data, junc_query):
        self.id = data['seq_id']
        self.v_fam = data['v_gene']['fam']
        self.v_gene = data['v_gene']['gene']
        self.v_all = data['v_gene']['all']
        self.j_gene = data['j_gene']['gene']
        self.j_all = data['j_gene']['all']
        self.junc = data[junc_query]
        self.junc_len = len(self.junc)
        self.muts = []
        if 'var_muts_nt' in data.keys():
            self.muts = data['var_muts_nt']

    def v_gene_string(self):
        return 'v{0}-{1}'.format(self.v_fam, self.v_gene)

    def v_fam_string(self):
        return 'v{0}'.format(self.v_fam)


def get_LD(i, j):
    '''Calculate sequence distance between a pair of Seq objects'''
    # pairwise2 is used to force 'gapless' distance when sequence pair is of the same length
    if i.junc_len == j.junc_len:
        identity = pairwise2.align.globalms(i.junc, j.junc, 1, 0, -50, -50, score_only=True, one_alignment_only=True)
        return i.junc_len - identity
    # Levenshtein distance is used for sequence pairs of different lengths
    else:
        return distance(i.junc, j.junc)


def vCompare(i, j):
    '''Calculate penalty for mismatches in Variable segment.'''
    if i.v_gene != j.v_gene:
        return 8
    if i.v_all != j.v_all:
        return 1
    return 0


def jCompare(i, j):
    '''Calculate penalty for mismatches in Joining segment.'''
    if i.j_gene != j.j_gene:
        return 8
    if i.j_all != j.j_all:
        return 1
    return 0


def sharedMuts(i, j):
    '''Calculate bonus for shared mutations.'''
    if i.id == j.id:
        return 0.0
    bonus = 0.0
    for mut in i.muts:
        if mut == '':
            continue
        if mut in j.muts:
            bonus += 0.35
    return bonus


def get_score(i, j=None):
    if j is None:
        i, j = i
    if i.id == j.id:
        return 0.0
    LD = get_LD(i, j)
    vPenalty = vCompare(i, j)
    jPenalty = jCompare(i, j)
    lenPenalty = math.fabs(i.junc_len - j.junc_len) * 2
    editLength = min(i.junc_len, j.junc_len)
    mutBonus = sharedMuts(i, j)
    if mutBonus > (LD + vPenalty + jPenalty):
        mutBonus = (LD + vPenalty + jPenalty - 0.001)  # distance values can't be negative
    return (LD + vPenalty + jPenalty + lenPenalty - mutBonus) / editLength


def make_iter(seqs, mode=1):
    for i, seq_i in enumerate(seqs):
        if mode == 1:
            for seq_j in seqs[i + 1:]:
                yield (seq_i, seq_j)
        else:
            yield (seq_i, seqs[i + 1:])


def get_scores_one_row(args):
    (seq_i, row_j) = args
    return np.array([get_score(seq_i, seq_j) for seq_j in row_j], dtype=default_dtype)


def build_condensed_matrix(seqs, mode=2):
    result = np.array([], dtype=default_dtype)
    p = Pool(processes=cpu_count())
    if mode == 1:
        n = len(seqs)
        #chunksize = 500000
        chunksize = int(n * (n - 1) / 2 / cpu_count() / 2)
        result_one = p.imap(get_score, make_iter(seqs, mode=1), chunksize=chunksize)
        result = np.array(list(result_one), dtype=default_dtype)
    else:
        result_one_row = p.imap(get_scores_one_row, make_iter(seqs, mode=2), chunksize=100)
        result = np.concatenate(list(result_one_row))
    #p.close()
    #p.join()
    return result


def build_cluster_dict(count, vh):
    clusters = {}
    for c in range(1, count):
        clusters["lineage_{0}_{1}".format(vh, str(c))] = []
    return clusters


def assign_seqs(flatCluster, clusters, input_seqs, vh):
    for s in range(len(flatCluster)):
        s_id = 'lineage_{0}_{1}'.format(vh, str(flatCluster[s]))
        clusters[s_id].append(input_seqs[s])
    return clusters


def make_clusters(con_distMatrix, input_seqs):
    vh = 'v0'
    #print 'clustering...'
    linkageMatrix = fc.linkage(con_distMatrix, method='average', preserve_input=False)
    flatCluster = fcluster(linkageMatrix, distance_cutoff, criterion='distance')
    del linkageMatrix
    #print 'building cluster dict...'
    clusters = build_cluster_dict(max(flatCluster) + 1, vh)
    #print 'assigning sequences to clusters...'
    clusters = assign_seqs(flatCluster, clusters, input_seqs, vh)
    return clusters


def write_output(outfile, data):
    with open(outfile, 'w') as out_f:
        for c in data.keys():
            rString = ''
            if len(data[c]) < 2:
                continue
            rString += '#{}\n'.format(c)
            for seq in data[c]:
                rString += '>{0}\n{1}\n'.format(seq.id, seq.junc)
            rString += '\n'
            out_f.write(rString)


def print_resource_usage():
    import resource
    usage = resource.getrusage(resource.RUSAGE_SELF)
    print('max_rss:', usage.ru_maxrss)


def get_memery_usage():
    import subprocess
    import os
    rss = subprocess.check_output('ps -p {} u'.format(os.getpid()), shell=True).decode('utf-8').split('\n')[1].split()[5]
    print('current_rss:', rss)


def analyze(infile, outfile=None, n=None):
    get_memery_usage()
    print_resource_usage()

    t00 = time.time()
    print("Loading input sequences...", end='')
    with open(infile) as in_f:
        seqs = json.load(in_f)
        if n:
            seqs = seqs[:n]
    seqs = [Seq(s, 'junc_aa') for s in seqs]
    print("done. [{}, {:.2f}s]".format(len(seqs), time.time() - t00))
    get_memery_usage()
    print_resource_usage()

    t0 = time.time()
    print("Calculating condensed distance matrix...", end='')
    con_distMatrix = build_condensed_matrix(seqs, mode=2)   # ####
    print("done. [{}, {:.2f}s]".format(con_distMatrix.shape, time.time() - t0))
    print("\tmin: {}, max: {}".format(con_distMatrix.min(), con_distMatrix.max()))
    get_memery_usage()
    print_resource_usage()

    t0 = time.time()
    print("Calculating clusters...", end='')
    clusters = make_clusters(con_distMatrix, seqs)
    print("done. [{}, {:.2f}s]".format(len(clusters), time.time() - t0))
    get_memery_usage()
    print_resource_usage()

    t0 = time.time()
    print ("Outputting clusters...", end='')
    outfile = outfile or '_clone.'.join(infile.rsplit('.', 1))
    write_output(outfile, clusters)
    print("done. {:.2f}s".format(time.time() - t0))

    print('=' * 20)
    print("Finished! Total time= {:.2f}s".format(time.time() - t00))
    get_memery_usage()
    print_resource_usage()

if __name__ == '__main__':
    try:
        infile = sys.argv[1]
    except:
        print("Usage: python clonify_contest.py <infile> [outfile]")
        sys.exit()
    try:
        outfile = sys.argv[2]
    except:
        outfile = None
    try:
        n = int(sys.argv[3])
    except:
        n = None

    analyze(infile, outfile, n)
