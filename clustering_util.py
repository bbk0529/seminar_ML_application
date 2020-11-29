
from pm4py.objects.log.importer.xes import factory as xes_import_factory
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.algo.discovery.heuristics import factory as heuristics_miner
from pm4py.evaluation.replay_fitness import factory as replay_factory

from pm4py.visualization.petrinet import factory as pn_vis_factory
from pm4py.visualization.heuristics_net import factory as hn_vis_factory
from pm4py.visualization.petrinet import visualizer as pn_visualizer

import numpy as np
from sklearn.cluster import KMeans
from discover_mr import discover_maximal_repeat

def mra_arr_generator(VARIANT) : 
    mra_arr = []
    for variant in VARIANT :
        v = discover_maximal_repeat(variant.split(','))
        mra_arr.append(v)
    return mra_arr    

def boa_arr_generator(VARIANT) : 
    boa_arr=[]
    for v in VARIANT :
        activities = v.split(',')
        boa_arr.append(activities)
    return boa_arr

def ngram_generator (arr, n) : 
    ngram_arr=[]
    for i in range(len(arr) - n +1 ) :
        ar = arr[i:i+n]
        ngram_arr.append(','.join(ar))
    return ngram_arr    


def ngram_arr_generator(VARIANT, n = 3) : 
    ngram_arr=[]
    for v in VARIANT :
        ngram = ngram_generator(v.split(','),n)
        ngram = list(set(ngram)) #to remove duplications
        ngram_arr.append(ngram)
    return ngram_arr


def CS_creator(VARIANT, type=['ngram','boa','mra'], n=3, k=3):
    if type=='ngram' : 
        kmeans = kmean_launcher (VARIANT, type=type, n=n, k=k) 
    elif type == 'boa' : 
        kmeans = kmean_launcher(VARIANT, type=type) 
    elif type == 'mra' : 
        kmeans = kmean_launcher(VARIANT, type=type) 

    VARIANT = np.array(VARIANT)
    idx, cnt = np.unique(np.array(kmeans.labels_), return_counts=True)
    CS=[]
    for i in idx : 
        CS.append(VARIANT[kmeans.labels_==i])
    return CS



def kmean_launcher (VARIANT, type=['ngram','boa','mra'], n = 3, k = 5) : 
    print(type)
    if type == 'boa' :
        arr = boa_arr_generator(VARIANT)
    elif type =='ngram' :
        arr = ngram_arr_generator(VARIANT,n)
    elif type =='mra' : 
        arr = mra_arr_generator(VARIANT)

    features = feature_extractor(arr)
    feature_vectors = fecture_vectors_creator(arr, features)

    data = np.array(feature_vectors)
    kmeans = KMeans(n_clusters = k, random_state=0).fit(data)

    return kmeans



def fecture_vectors_creator (arr, features)  : 
    feature_vectors = []
    for a in arr : 
        feature = [0] * len(features)
        for n in a : 
            feature[features.index(n)] = 1
        feature_vectors.append(feature)
    return feature_vectors


def feature_extractor (arr) :
    features = []
    for g in arr : 
        features += g
    features=list(set(features))
    return features


def read_xes(filename, p=1) : 
    '''
    read event log in xes format 
        input   filename, percentage
        output  log object, variants_count
    
    filename = filename in xes format
    p = percentage of traces % to exploit from the log
    '''

    log = xes_import_factory.apply(filename)
    log = variants_filter.filter_log_variants_percentage(log, percentage=p) 
    variants = variants_filter.get_variants(log)
    VARIANT = list(variants.keys())
   
    print(
        'length of trace', len(log),
        '\nlength of event', sum(len(trace) for trace in log),
        '\nnumber of variants : {}'.format(len(VARIANT))
    )
    return log, VARIANT

def fit_check(log :list, C :list) -> float: 
    log = variants_filter.apply( # get the log containing variants in C 
            log, 
            [c for c in C]  
    ) 
    net, im, fm = heuristics_miner.apply(log)
    fit = replay_factory.apply(log, net, im, fm )

    try :
        fit['averageFitness']  
    
    except : 
        print("*************************[ERROR] look_ahead_fit['averageFitness'] does not exist, instead {}".format(fit))
        return 0
    
    return fit['averageFitness']


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





def visualization_total(log, VARIANT, CS, freq_check = False) :
    print("visualization of VARIANT") 
    visualization(log, VARIANT, True, False)
    
    print("visualization of each cluster in CS")
    for cs in CS : 
        print(len(cs))
        if freq_check : 
            print(fit_check(log, cs))
        visualization(log, cs, True, False)



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
