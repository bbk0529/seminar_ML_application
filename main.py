

from evaluation import *
import time
import datetime
import sys


if __name__ == "__main__":
    k = 4
    output = False
    visual = False
    w = 0.25
    tf = 1
    mcs = 0.25
    N = 1
    p = 1

    filename = sys.argv[1]
    log, VARIANT = read_xes(filename)
    start_act = time.time()
    if output:
        print("* active clustering started, it may take some time to finish. to see the progress, please use output = True")
    CS_act = A_clustering(
        log, VARIANT,
        w=w,  tf=tf, nb_clus=k, mcs=mcs,
        N=N,
        output=output,
        visual=visual
    )
    end_act = time.time()

    start_boa = time.time()
    CS_boa = CS_creator(VARIANT, type='boa', k=k)
    end_boa = time.time()

    start_mra = time.time()
    CS_mra = CS_creator(VARIANT, type='mra', k=k)
    end_mra = time.time()

    start_ngram = time.time()
    CS_ngram = CS_creator(VARIANT, type='ngram', n=3, k=k)
    end_ngram = time.time()

    time_act = end_act - start_act
    time_boa = end_boa - start_boa
    time_mra = end_mra - start_mra
    time_ngram = end_ngram - start_ngram
    time_list = [time_act, time_boa, time_mra, time_ngram]

    result = []
    for type in [CS_act, CS_boa, CS_mra, CS_ngram]:
        result.append(quality_measure(log, type))
    result = np.array(result)
    idx = filename.rfind('/', )
    filename = filename[idx+1:-4]

    now = str(datetime.datetime.now().date())
    pickle.dump(CS_act, open("./pickles/CS_act_" +
                             filename + "_" + now + ".p", "wb"))
    pickle.dump(CS_boa, open("./pickles/CS_boa_" +
                             filename + "_" + now + ".p", "wb"))
    pickle.dump(CS_mra, open("./pickles/CS_mra_" +
                             filename + "_" + now + ".p", "wb"))
    pickle.dump(CS_ngram, open(
        "./pickles/CS_ngram_" + filename + "_" + now + ".p", "wb"))

    pickle.dump(result, open(
        "./pickles/result_" + filename + "_" + now + ".p", "wb"))

    pickle.dump(time_list, open(
        "./pickles/runtime_" + filename + "_" + now + ".p", "wb"))
