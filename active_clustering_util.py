from pm4py.statistics.traces.log import case_statistics
from discover_mr import discover_maximal_repeat
from pm4py.algo.filtering.log.variants import variants_filter

from clustering_util import *

import importlib
import clustering_util

importlib.reload(clustering_util)

def W_creater (log, R, w, output=False):
    
    W = []
    log = variants_filter.apply(log,R)
    target_size = len(log) * w  #it determines the size of W 
    variant = case_statistics.get_variant_statistics(log)
    # variant = sorted(variant, key=lambda x: x['count'], reverse=True)
    if output : print("="*100, "\nW creater called with w : {} and target size {}\n".format(w, target_size))
    W_size = 0
    for v in variant : 
        W_size += v['count']
        W.append(v['variant'])
        if output : 
            print(
                "\t\t{}___added  // {} out of {}  // total size : {}".
                format(v['variant'][:60], W_size, target_size, len(log))
            )
        
        if W_size > target_size : 
            break
            
    if output : 
        print("W creater END")
        print("="*100)
    return W



def min_distance_seeker(dpi, C) :
    import sys
    min_dist = sys.maxsize #Instead of inf. 
    for c in C : 
        if min_dist > dist_btw_traces(dpi,c) : 
            min_dpi = c
    return min_dpi


def dpi_finder(C,W, output=False) :
    min_avg_dist = 9999 #arbitrary big number / dist cannot be larger than the number of all activities 
    for w in W : 
        sum_dist = 0
        w_mr = discover_maximal_repeat(w.split(','))
        
        for c in C :
            c_mr = discover_maximal_repeat(c.split(',')) 
            if output : print("\t maximal repeat in w:{} c:{} ".format(w_mr, c_mr))
            sum_dist += dist_btw_set(w_mr,c_mr)
        
        if sum_dist / len(C) < min_avg_dist : 
            min_avg_dist = sum_dist / len(C)
            cur_dpi = w
            if output : print("\tUPDATED cur_dpi in the loop|",cur_dpi[:40], "\t",min_avg_dist)


    print("\n * Selected dpi via dpi_finder() :\n\t{}... with dist {}".format(cur_dpi[:60],min_avg_dist))
    return cur_dpi




def look_ahead(log :list, C, R) : 
    print("\n * Look_ahead()")
    C_log = variants_filter.apply(log, C) 
    net, im, fm = heuristics_miner.apply(C_log)
    for i, r in enumerate(R) : 
        if i%10 == 0 :
            print("\t = {} dpi(s) checked".format(i))
        r_log = variants_filter.apply(log, r) 
        fit = replay_factory.apply(r_log, net, im, fm )

        try : 
            if fit['averageFitness'] == 1 : 
                print("\tFound a perfect fitness - {}".format(r))
                R.remove(r)
                C.append(r)
        except : 
            print("******************[ERROR] _ look_ahead_fit['averageFitness'] does not exist, instead {}".format(fit))
            continue
    return C, R



