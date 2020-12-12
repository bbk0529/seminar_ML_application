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


def W_creater(log, R, w, output=False):

    W = []
    log = variants_filter.apply(log, R)
    target_size = len(log) * w  # it determines the size of W
    variant = case_statistics.get_variant_statistics(log)
    # variant = sorted(variant, key=lambda x: x['count'], reverse=True)
    if output:
        print("="*100, "\nW creater called with w : {} and target size {}\n".format(w, target_size))
    W_size = 0
    for v in variant:
        W_size += v['count']
        W.append(v['variant'])
        # if output:
        #     print(
        #         "\t\t{}___added with size {} // {} out of {}  // total size : {}".
        #         format(v['variant'][:60], v['count'],
        #                W_size, target_size, len(log))
        #     )

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


def dpi_finder(C, W, output=False):
    # arbitrary big number / dist cannot be larger than the number of all activities
    min_avg_dist = 9999
    for w in W:
        sum_dist = 0
        w_mr = discover_maximal_repeat(w.split(','))

        for c in C:
            c_mr = discover_maximal_repeat(c.split(','))
            # if output:
            #     print("\t original w:{} c:{} ".format(w, c))
            #     print(
            #         "\t\t correspnding maximal repeat in w:{} c:{} ".format(w_mr, c_mr))
            sum_dist += dist_btw_set(w_mr, c_mr)

        if sum_dist / len(C) < min_avg_dist:
            min_avg_dist = sum_dist / len(C)
            cur_dpi = w
            if output:
                print("\tUPDATED cur_dpi in the loop|",
                      cur_dpi[:40], "\t", min_avg_dist)

    if output:
        print(
            "\n * Selected dpi via dpi_finder() :\n\t{}... with dist {}".format(cur_dpi[:60], min_avg_dist))
    return cur_dpi


def look_ahead(log: list, C, R, output=False):
    if output:
        print("\n * Look_ahead()")
    C_log = variants_filter.apply(log, C)
    net, im, fm = heuristics_miner.apply(C_log)
    # net, im, fm = inductive_miner.apply(C_log)
    for i, r in enumerate(R):
        if i % 10 == 0:
            if output:
                print("\t = {} dpi(s) checked".format(i))
        r_log = variants_filter.apply(log, r)
        fit = replay_fitness_evaluator.apply(
            r_log, net, im, fm, variant=replay_fitness_evaluator.Variants.TOKEN_BASED)

        if fit == 1:
            if output:
                print("\tFound a perfect fitness - {}".format(r))
            R.remove(r)
            C.append(r)
    return C, R
