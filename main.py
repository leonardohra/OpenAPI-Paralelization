import os
import pickle
from Gensim_Model import *
import time

import threading
import multiprocessing

def load_list(file):
    itemlist = []

    with open (file, 'rb') as fp:
        itemlist = pickle.load(fp)

    return itemlist

def get_summary_and_description_endpoints(data_list):
    summaries = []
    descriptions = []
    summaries_append_description = []
    for api, endpoints in data_list.items():
        for endpoint, components in endpoints.items():
            if('description' in components.keys() and 'summary'  in components.keys()):
                summaries.append(components['summary'])
                descriptions.append(components['description'])
                summaries_append_description.append(components['summary'] + ' ' + components['description'])

    return summaries, descriptions, summaries_append_description

def set_up():
    source = './APIs/'
    list_file = './data.txt'
    file_exists = os.path.isfile(list_file)
    data_list = []

    if(file_exists):
        data_list = load_list(list_file)
    else:
        data_list = generate_list(list_file, source)

    print ('Number of APIs: ' + str(len(data_list)))

    sum, des, s_p_d = get_summary_and_description_endpoints(data_list)
    output_folder = './results/'
    outfile = 'run'
    data_list_sum = sum
    data_list_des = des
    data_list_spd = s_p_d
    qtt_topics_n = range(2, 4)
    coherence_t = [ CoherenceType.C_V ]
    topns_n = [5, 10]

    return topns_n, coherence_t, qtt_topics_n, data_list_spd, data_list_des, data_list_sum, outfile, output_folder


def main_normal():
    print("Without using paralelization or threads or anything")
    now = time.time()

    topns_n, coherence_t, qtt_topics_n, data_list_spd, data_list_des, data_list_sum, outfile, output_folder = set_up()

    #gen_mod_sum = Gensim_Model(data_list_spd)
    #gen_mod_sum.pre_processing()
    #gen_mod_sum.model_construction_evaluation_topn(output_folder, outfile + '_spd', qtt_topics=qtt_topics_n, coherence_types = coherence_t, topns= topns_n)

    gen_mod_sum = Gensim_Model(data_list_sum)
    gen_mod_sum.pre_processing()
    gen_mod_sum.model_construction_evaluation_topn(output_folder, outfile + '_norm_sum', qtt_topics=qtt_topics_n, coherence_types = coherence_t, topns= topns_n)

    gen_mod_des = Gensim_Model(data_list_des)
    gen_mod_des.pre_processing()
    gen_mod_des.model_construction_evaluation_topn(output_folder, outfile + '_norm_des', qtt_topics=qtt_topics_n, coherence_types = coherence_t, topns= topns_n)

    then = time.time()
    print('time it took: {}'.format(then - now))

def mod_const(dataset, out_fo, out_fi, qtt_topics_n, coherence_t, topns_n):
    print("Doing {}".format(out_fi))
    gen_mod = Gensim_Model(dataset)
    gen_mod.pre_processing()
    gen_mod.model_construction_evaluation_topn(out_fo, out_fi, qtt_topics=qtt_topics_n, coherence_types = coherence_t, topns= topns_n)
    return gen_mod

def main_threading():
    print("Using Threading lib")
    now = time.time()

    topns_n, coherence_t, qtt_topics_n, data_list_spd, data_list_des, data_list_sum, outfile, output_folder = set_up()

    #t1 = threading.Thread(target=mod_const, args=[data_list_spd, output_folder, outfile + '_spd', qtt_topics_n, coherence_t, topns_n])
    t2 = threading.Thread(target=mod_const, args=[data_list_des, output_folder, outfile + '_thr_des', qtt_topics_n, coherence_t, topns_n])
    t3 = threading.Thread(target=mod_const, args=[data_list_sum, output_folder, outfile + '_thr_sum', qtt_topics_n, coherence_t, topns_n])

    #t1.start()
    t2.start()
    t3.start()

    #t1.join()
    t2.join()
    t3.join()

    then = time.time()
    print('time it took: {}'.format(then - now))

def main_multiprocessing():
    print("Using Multiprocessing lib")
    now = time.time()

    topns_n, coherence_t, qtt_topics_n, data_list_spd, data_list_des, data_list_sum, outfile, output_folder = set_up()

    #p1 = multiprocessing.Process(target=mod_const, args=[data_list_spd, output_folder, outfile + '_spd', qtt_topics_n, coherence_t, topns_n])
    p2 = multiprocessing.Process(target=mod_const, args=[data_list_des, output_folder, outfile + '_multp_des', qtt_topics_n, coherence_t, topns_n])
    p3 = multiprocessing.Process(target=mod_const, args=[data_list_sum, output_folder, outfile + '_multp_sum', qtt_topics_n, coherence_t, topns_n])

    #p1.start()
    p2.start()
    p3.start()

    #p1.join()
    p2.join()
    p3.join()

    then = time.time()
    print('time it took: {}'.format(then - now))


if __name__ == "__main__":
    main_normal()
    #668.4853866100311 seconds
    main_threading()
    #421.41410541534424 seconds
    main_multiprocessing()
    #386.0526068210602 seconds
