import numpy as np
import pickle
from pm4py.algo.filtering.log.variants import variants_filter

from pm4py.evaluation.replay_fitness import evaluator as replay_fitness_evaluator
from pm4py.evaluation.precision import evaluator as precision_evaluator
from pm4py.evaluation.generalization import evaluator as generalization_evaluator
from pm4py.evaluation.simplicity import evaluator as simplicity_evaluator
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner


import matplotlib
import matplotlib.pyplot as plt

from active_clustering import *
from clustering_util import *


def visualize_evaluation(result):

    labels = ['Fit', 'Pre', 'gen', 'simp']

    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 3/2 * width, result[0], width, label='Active Tracing')
    rects2 = ax.bar(x - 1/2 * width,
                    result[1], width, label='boa', color='gray')
    rects3 = ax.bar(x + 1/2 * width,
                    result[2], width, label='mra', color='black')
    rects4 = ax.bar(x + 3/2 * width,
                    result[3], width, label='n_gram', color='lightgrey')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    plt.show()


def quality_measure(log, CS):

    # fitness, prec, gen, simp, weighted by # traces

    eval = []
    for cs in CS:
        l = variants_filter.apply(log, cs)
        eval.append(evaluation_w_hm(l))

    DATA = np.array(eval)
    print(DATA)
    metrics = []
    for i in range(1, DATA.shape[1]):
        metrics.append(
            sum(
                DATA[:, 0] * DATA[:, i]
            )/sum(DATA[:, 0])
        )
    print(
        "fitness:{}, prec:{}, gen:{}, simp:{}, weighted by # traces".
        format(metrics[0], metrics[1], metrics[2], metrics[3])
    )
    return metrics


def evaluation_w_hm(log):
    net, im, fm = inductive_miner.apply(log)

    fitness = replay_fitness_evaluator.apply(
        log, net, im, fm, variant=replay_fitness_evaluator.Variants.TOKEN_BASED)['log_fitness']
    prec = precision_evaluator.apply(
        log, net, im, fm, variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
    gen = generalization_evaluator.apply(log, net, im, fm)
    simp = simplicity_evaluator.apply(net)

    return [len(log), fitness, prec, gen, simp]


def total_evaluation():
    log = pickle.load(open('log.p', 'rb'))
    CS_act = pickle.load(open("CS_act.p", 'rb'))
    CS_boa = pickle.load(open("CS_boa.p", 'rb'))
    CS_mra = pickle.load(open("CS_mra.p", 'rb'))
    CS_ngram = pickle.load(open("CS_ngram.p", 'rb'))
    result = []

    for type in [CS_act, CS_boa, CS_mra, CS_ngram]:
        result.append(quality_measure(log, type))
    result = np.array(result)

    return result


def total_clustering(
    filename, k, output=False, visual=False,
    w=1,  tf=0.99, mcs=0.25,
    N=1,
    p=1
):
    # file name are hard-coded here. to be soft coded if required.
    log, VARIANT = read_xes(filename, p=p)
    pickle.dump(log, open('log.p', 'wb'))
    pickle.dump(VARIANT, open('VARIANT.p', 'wb'))
    if output:
        print("* active clustering started, it may take some time to finish. to see the progress, please use output = True")

    CS_act = A_clustering(
        log, VARIANT,
        w=w,  tf=tf, nb_clus=k, mcs=mcs,
        N=N,
        output=output,
        visual=visual
    )

    if output:
        print("* active clustering finished")

    if output:
        print("* BOA clustering finished")
    CS_boa = CS_creator(VARIANT, type='boa', k=k)
    if output:
        print("* BOA clustering finished")

    if output:
        print("* MRA clustering finished")
    CS_mra = CS_creator(VARIANT, type='mra', k=k)
    if output:
        print("* MRA clustering finished")

    if output:
        print("* ngram clustering finished")
    CS_ngram = CS_creator(VARIANT, type='ngram', n=3, k=k)
    if output:
        print("* ngram clustering finished")

    pickle.dump(CS_act, open("CS_act.p", "wb"))
    pickle.dump(CS_boa, open("CS_boa.p", "wb"))
    pickle.dump(CS_mra, open("CS_mra.p", "wb"))
    pickle.dump(CS_ngram, open("CS_ngram.p", "wb"))
