from pm4py.evaluation.replay_fitness import evaluator as replay_fitness_evaluator
import clustering_util
import active_clustering_util
import importlib
from active_clustering_util import *
from clustering_util import *
import warnings
import numpy as np
warnings.filterwarnings("ignore")


importlib.reload(active_clustering_util)
importlib.reload(clustering_util)


def clustering(C, I, R, log, mcs, tf, w, visual=False, output=False):
    if output:
        if output:
            print("\nClustering() is called. mcs:{}, tf:{}, w:{}".format(mcs, tf, w))
    while (len(R) > 0 and set(R) != set(I)):  # line 8
        if output:
            print("-"*100)
        if output:
            print("START OF LOOP with cur_dpi")

        if w:
            W = W_creater(log, list(set(R) - set(I)), w, output)
        else:  # if w is 0, frequency based selective search
            if output:
                print(
                    "\n * As w is set 0, it searches the most frequent dpi from R_ActiTracC_freq")
            W = [R[0]]  # W is just a single dpi with the highest freuqency

        if (len(C) == 0):  # if C is empty set
            cur_dpi = R[0]  # R is already sorted in increasing order.
            if output:
                print(
                    "\n * C is empty set - size of |C|:{} ->  R[0] or  is to be added.\n".format(len(C), len(W)))

        elif len(C) > 0 and (len(W) == 1):
            cur_dpi = W[0]
            if output:
                print(
                    "\n * C is not empty set, but |W| = 1. |C|:{}, |W|:{} ->  W[0] or  is to be added.\n".format(len(C), len(W)))

        else:
            if output:
                print(
                    "\n * C is not empty set and W is larger than 1, so w in W to be selected with min_dist")
            cur_dpi = dpi_finder(C, W, output=output)

        if output:
            print("\tcur_dpi = R[0] {}...\n\n".format(cur_dpi[:40]))
        # print("\n * Fitness check to be done with cur_dpi\n\t {}...".format(cur_dpi[:80]))
        if output:
            print("\n * Fitness check to be done with cur_dpi\n\t {}...".format(cur_dpi))

        fit = fit_check_w_HM(log, cur_dpi, C)
        if fit >= tf:
            R.remove(cur_dpi)
            C.append(cur_dpi)  # added to C
            if output:
                print(
                    "\n * CASE of fit {} >= {} tf -> Cur_dpi is added to cluster C & removed from R\n\t" .
                    format(round(fit, 2), tf)
                )

            if visual:
                visualization(log, C)

        else:  # if fit < tf
            if output:
                print(
                    "\n * CASE of fit {} < {} tf -> fitness dropped than the tf".format(fit, tf))
            C_size = len(variants_filter.apply(log, C))
            R_size = len(variants_filter.apply(log, R))

            if C_size >= mcs * R_size:
                if output:
                    print(
                        "\t - CASE of |C| {} >= {} mcs * |R| -> look_ahead is called, then this clustering is completed".
                        format(C_size, mcs * R_size)
                    )
                C, R = look_ahead(log, C, R, output=output)

                if output:
                    print("\n * Clustering completed")
                for c in C:
                    if output:
                        print("\t\t{}".format(c))
                if visual:
                    visualization(log, C)
                return C, R

            else:
                if output:
                    print(
                        "\t - CASE of |C|{} <= {} mcs*|R| -> still it need more trace, cur_dpi added to I and the loop continues ".
                        format(C_size, mcs * R_size)
                    )
                I.append(cur_dpi)
                I = list(set(I))
        if output:
            print(
                "\nEND OF LOOP with cur_dpi____fit : {} / size of C: {} / size of R: {} / size of I: {}"
                .format(round(fit, 2), len(C), len(R), len(I))
            )
        if output:
            print("\n")

    return C, R


def residual_trace_resolution(R, CS, log, output=False):
    if output:
        print("STEP 3 : residual trace resolution ahead step start")
    # LOOK AHEAD STEPS
    for no, r in enumerate(R):
        # print("\n{}".format(r))
        fit_max = 0
        fit_max_idx = -1
        for i in range(len(CS)):
            C_log = variants_filter.apply(log, CS[i])
            net, im, fm = heuristics_miner.apply(C_log)
            r_log = variants_filter.apply(log, r)
            fit = replay_fitness_evaluator.apply(
                r_log, net, im, fm, variant=replay_fitness_evaluator.Variants.TOKEN_BASED)['log_fitness']
            if fit_max < fit:
                fit_max = fit
                fit_max_idx = i
        if output:
            print("{} out of {} is added to {} cluster with fitness{} : {}".format(
                no, len(R), fit_max_idx, round(fit_max, 2), r))
        CS[i].append(r)
    return CS


def A_clustering(
    log, VARIANT,
    w=1,  tf=0.99, nb_clus=5, mcs=0.25,
    N=1,
    output=False,
    visual=False

):

    R = VARIANT.copy()
    CS = []

    for i in range(nb_clus-N):
        if output:
            print("*"*100)
        if output:
            print("START OF No. {} CLUSTERING\n".format(i))
        C = []
        I = []
        C, R = clustering(
            C, I, R,
            log, mcs, tf, w,
            visual=visual,
            output=output
        )
        CS.append(C)

        R_size = len(variants_filter.apply(log, R))
        log_size = len(log)
        progress = 1 - round(R_size / log_size, 2)

        if output:
            print(
                "COMPLETION OF SINGLE CLUSTERING {} been clustered ({} out of {}) // Remaining # traces {}".
                format(progress, log_size - R_size, log_size, len(R))
            )
    if output:
        print("COMPLETION OF WHOLE CLUSTERING\n")

    if N:
        if output:
            print(
                "STEP 3_ since N = 1, all the remaining traces are collected into new single cluster")
        CS.append(R)
    else:
        if output:
            print(
                "STEP 3_ since N = 0, all the remaining traces are moved to the most suitable clusters")
        CS = residual_trace_resolution(R, CS, log)
    return CS
