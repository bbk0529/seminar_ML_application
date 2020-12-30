from pm4py.statistics.traces.log import case_statistics
from discover_mr import discover_maximal_repeat
from pm4py.algo.filtering.log.variants import variants_filter

from pm4py.evaluation.replay_fitness import evaluator as replay_fitness_evaluator
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner

from clustering_util import *

import importlib
import clustering_util

importlib.reload(clustering_util)


def MR_creator(VARIANT):
    '''
    INPUT : VARIANT
    output : MAXIMAL REPEAT in the list

    '''
    concatenated_log = []
    for trace in VARIANT:
        sequence = [event for event in trace.split(',')]
        sequence.append('|')
        concatenated_log += sequence

    return discover_maximal_repeat(concatenated_log)


def MRA_creator(mr):
    mra = set()
    for r in mr:
        for a in r.split(','):
            mra.add(a)
    return mra


def W_creater(log, R, w, output=False):

    W = []
    log = variants_filter.apply(log, R)
    target_size = len(log) * w  # it determines the size of W
    variant = case_statistics.get_variant_statistics(log)
    variant = sorted(variant, key=lambda x: x['count'], reverse=True)
    if output:
        print("="*100, "\nW creater called with w : {} and target size {}\n".format(w, target_size))
    W_size = 0
    for v in variant:
        W_size += v['count']
        W.append(v['variant'])
        if output:
            print(
                "\t\t{}___added with size {} // {} out of {}  // total size : {}".
                format(v['variant'][:60], v['count'],
                       W_size, target_size, len(log))
            )

        if W_size > target_size:
            break

    if output:
        print("W creater END with its size: {}".format(len(W)))
        print("="*100)
    return W


def min_distance_seeker(dpi, C):
    import sys
    min_dist = sys.maxsize  # Instead of inf.
    for c in C:
        if min_dist > dist_btw_traces(dpi, c):
            min_dpi = c
    return min_dpi


def dpi_finder(C, W, mra, output=False):
    C_in_mra = []
    for v in C:
        C_in_mra.append(mra.intersection(set(v.split(','))))

    W_in_mra = []
    for v in W:
        W_in_mra.append(mra.intersection(set(v.split(','))))

    dist_mat = np.zeros((len(W_in_mra), len(C_in_mra)))
    for w_idx, w in enumerate(W_in_mra):
        for c_idx, c in enumerate(C_in_mra):
            dist_mat[w_idx, c_idx] = dist_btw_set(w, c)
    idx = np.argmin(np.sum(dist_mat, axis=1))
    print(np.sum(dist_mat, axis=1))
    return W[idx]


def look_ahead(log: list, C, R, output=False):
    if output:
        print("\n * Look_ahead()")
    C_log = variants_filter.apply(log, C)
    net, im, fm = heuristics_miner.apply(C_log)
    # net, im, fm = inductive_miner.apply(C_log)
    for i, r in enumerate(R):
        if i % 10 == 0:
            print("\t = {} dpi(s) checked".format(i))
        r_log = [variants_filter.apply(log, [r])[0]]
        fit = replay_fitness_evaluator.apply(
            r_log, net, im, fm, variant=replay_fitness_evaluator.Variants.TOKEN_BASED)

        if fit == 1:
            print("fitness:", fit)
            if output:
                print("\tFound a perfect fitness - {}".format(r))
            R.remove(r)
            C.append(r)
    return C, R
