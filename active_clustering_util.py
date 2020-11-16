import warnings
import numpy as np
warnings.filterwarnings("ignore", category=DeprecationWarning) 


from pm4py.algo.discovery.heuristics import factory as heuristics_miner
from pm4py.evaluation.replay_fitness import factory as replay_factory


from pm4py.algo.discovery.heuristics import factory as heuristics_miner
from pm4py.evaluation.replay_fitness import factory as replay_factory
from pm4py.objects.log.importer.xes import factory as xes_import_factory
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.statistics.traces.log import case_statistics


from pm4py.visualization.petrinet import factory as pn_vis_factory
from pm4py.visualization.heuristics_net import factory as hn_vis_factory
from pm4py.visualization.petrinet import visualizer as pn_visualizer


from discover_mr import discover_maximal_repeat


def read_xes(filename) : 
    log = xes_import_factory.apply(filename)

    print(
        'length of trace', len(log),
        '\nlength of event', sum(len(trace) for trace in log)
    )
    variants = variants_filter.get_variants(log)
    variants_count= case_statistics.get_variant_statistics(log)
    
    return log, variants_count


def add_frequency_into_variants_count(variants_count) : 
    #total sum
    s = 0 
    for i in range(len(variants_count)) : 
        s += variants_count[i]['count'] 

    #frequency 
    for i in range(len(variants_count)) : 
        variants_count[i]['freq']  = variants_count[i]['count'] / s
        if i == 0 : 
            variants_count[i]['acc_freq']  = variants_count[i]['freq']
        else : 
            variants_count[i]['acc_freq'] = variants_count[i-1]['acc_freq'] + variants_count[i]['freq']
    return variants_count




def dist_btw_set(trace1,trace2, output=False) : 
    A = set(trace1)
    B = set(trace2)
    if output : 
        print("\nA SET", A)
        print("\nB SET", B)
        print("\nUNION", A.union(B))
        print("\nINTERSECTION", A.intersection(B))
        print("\nDIFFERENCE", A.union(B) - A.intersection(B))
    return(len(A.union(B) - A.intersection(B)))




def fit_check_w_HM(log :list, cur_dpi :list, C :list) -> float: 
    log = variants_filter.apply( # get the log containing variants in C 
            log, 
            [c for c in C + [cur_dpi]]  
    ) 
    net, im, fm = heuristics_miner.apply(log)
    fit = replay_factory.apply(log, net, im, fm )

    try :
        fit['averageFitness']  
    
    except : 
        print("*************************[ERROR] look_ahead_fit['averageFitness'] does not exist, instead {}".format(fit))
        return 0
    
    return fit['averageFitness']


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

def extract_activities(R) :
    activities=[]
    for r in R : 
        activities +=  (r.split(','))
        activities = list(set(activities)) #set of all activities in the log
    return activities


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



def clustering(C, I, R, log, mcs, tf, w, visual=False, output=False) : 
    print("\nClustering() is called. mcs:{}, tf:{}, w:{}".format(mcs,tf,w))
    while (len(R)>0 and R != I) :  #line 8 
        print("-"*100)
        print("START OF LOOP with cur_dpi")

        if w : 
            W = W_creater(log, list(set(R) - set(I)), w, output)
        else : #if w is 0, frequency based selective search 
            print("\n * As w is set 0, it searches the most frequent dpi from R_ActiTracC_freq")
            W = [R[0]] #W is just a single dpi with the highest freuqency
        
        if (len(C) ==0 or len(W) == 1) : #if C is empty set
            print("\n * C is empty set or |W| = 1. |C|:{}, |W|:{} ->  R[0] is to be added.\n".format(len(C), len(W)))
            cur_dpi = R[0] #R is already sorted in increasing order. 
            if output : print("\tcur_dpi = R[0] {}...\n\n".format(cur_dpi[:40]))
        else :
            print("\n * C is not empty set, so w in W to be selected with min_dist")
            cur_dpi = dpi_finder(C,W)

        
        print("\n * Fitness check to be done with cur_dpi\n\t {}...".format(cur_dpi[:80]))
        
        fit = fit_check_w_HM(log, cur_dpi, C)
        if fit >= tf : 
            R.remove(cur_dpi)
            C.append(cur_dpi) # added to C
            print("\n * CASE of fit {} >= {} tf -> Cur_dpi is added to cluster C & removed from R\n\t" .format(fit,tf))
            
            if visual : visualization(log, C)
       
        else : # if fit < tf
            print("\n * CASE of fit {} < {} tf -> fitness dropped than the tf".format(fit, tf) )
            if len(C) >= mcs * len(R) :
                print(
                    "\t - CASE of |C| {} >= {} mcs * |R| -> look_ahead is called, then this clustering is completed".
                    format( len(C), mcs * len(R)) 
                )
                C,R = look_ahead(log,C,R)
                
                print("\n * Clustering completed")
                for c in C : 
                    print("\t\t{}".format(c))
                visualization(log, C)
                return C,R
            
            else :
                print("\t - CASE of |C|{} <= {} mcs*|R| -> still it need more trace, cur_dpi added to I and the loop continues ".format(len(C), mcs*len(R)))
                I.append(cur_dpi)
         
        print(
            "\nEND OF LOOP with cur_dpi____fit : {} / size of C: {} / size of R: {} / size of I: {}"
            .format(fit, len(C), len(R), len(I))
        )
        print("\n")
        
    return C,R

def visualization (log, C, petrinet=True, heu_net = False) :
    if petrinet : 
        net, im, fm = heuristics_miner.apply(variants_filter.apply(log, C))
        gviz = pn_vis_factory.apply(net, im, fm)
        pn_vis_factory.view(gviz)

    if heu_net :
        heu_net = heuristics_miner.apply_heu(variants_filter.apply(log, C))
        gviz = hn_vis_factory.apply(heu_net)
        hn_vis_factory.view(gviz)

    # if not filename : 
    #     parameters = {pn_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "jpg"}
    #     gviz = pn_visualizer.apply(net, parameters=parameters)
    #     pn_visualizer.save(gviz, filename + '.jpg')