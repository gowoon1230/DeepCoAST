import math
import sys
import os
from os import listdir
import time
import random
import numpy as np
from tqdm import tqdm
import pickle
import argparse
# This file extracts the features for the different attacks (parallel version)
# TODO: one file for parallel and single thread

def get1_DTAM(sizes1, sizes2, times1, times2):
    max_matrix_len = 1800 ##
    maximum_load_time = 80 ##

    feature_seq1 = [[0 for _ in range(max_matrix_len)], [0 for _ in range(max_matrix_len)]] # 2-DTAM
    for i in range(0, len(sizes1)):
        if sizes1[i] > 0:
            if times1[i] >= maximum_load_time: feature_seq1[0][-1] += 1
            else:
                idx = int(times1[i] * (max_matrix_len - 1) / maximum_load_time)
                feature_seq1[0][idx] += 1
        elif sizes1[i] < 0:
            if times1[i] >= maximum_load_time: feature_seq1[1][-1] += 1
            else:
                idx = int(times1[i] * (max_matrix_len - 1) / maximum_load_time)
                feature_seq1[1][idx] += 1
    feature_seq1 = np.array(feature_seq1[0]+feature_seq1[1]) # 1-DTAM

    feature_seq2 = [[0 for _ in range(max_matrix_len)], [0 for _ in range(max_matrix_len)]] # 2-DTAM
    for i in range(0, len(sizes2)):
        if sizes2[i] > 0:
            if times2[i] >= maximum_load_time: feature_seq2[0][-1] += 1
            else:
                idx = int(times2[i] * (max_matrix_len - 1) / maximum_load_time)
                feature_seq2[0][idx] += 1
        if sizes2[i] < 0:
            if times2[i] >= maximum_load_time: feature_seq2[1][-1] += 1
            else:
                idx = int(times2[i] * (max_matrix_len - 1) / maximum_load_time)
                feature_seq2[1][idx] += 1
    feature_seq2 = np.array(feature_seq2[0]+feature_seq2[1]) # 1-DTAM
    return feature_seq1, feature_seq2

def getDirection(sizes1, sizes2, times1, times2):
    feature_seq1 = np.array([np.sign (size) for size in sizes1])
    feature_seq2 = np.array([np.sign (size) for size in sizes2])
    return feature_seq1, feature_seq2

def getICD(sizes1, sizes2, times1, times2):
    feature_seq1 = [times1[i]-times1[i-1] for i in range(1, len(times1))]
    feature_seq2 = [times2[i]-times2[i-1] for i in range(1, len(times2))]
    feature_seq1.insert(0, 0)
    feature_seq2.insert(0, 0)
    return feature_seq1, feature_seq2

def getICDS(sizes1, sizes2, times1, times2):
    feature_seq1 = [times1[i]-times1[i-1] for i in range(1, len(times1))] + sizes1
    feature_seq2 = [times2[i]-times2[i-1] for i in range(1, len(times2))] + sizes2
    feature_seq1.insert(0, 0)
    feature_seq2.insert(0, 0)
    return feature_seq1, feature_seq2

def getTikTok(sizes1, sizes2, times1, times2):
    feature_seq1 = np.array([np.sign(sizes1[i]) * abs(times1[i]) for i in range(len(times1))])
    feature_seq2 = np.array([np.sign(sizes2[i]) * abs(times2[i]) for i in range(len(times2))])
    return feature_seq1, feature_seq2

def get_exp(avg_t, dist):
    if type(avg_t) is int and avg_t == 1:
        # no exponential if value is 1
        return 1
    # exponentional time
    if dist == "exp":
        return random.expovariate(float(1)/avg_t)
    else:  # fixed
        return avg_t

def write_features(directory, fname, p_in, p_out, p_dist, is_train, original_probas, FEATURE):
    # set param by DeepCoFFEA (Oh et al.)
    num_consecutive_packets_in = 20
    num_consecutive_packets_out = 20
    change_times = 0
    change_packets=0
    if change_times == 0 and change_packets == 0:
        change_dist = "fixed"
    num_conspack_dist = "exp"
    selected_network = [] # Which Network is selected: 1 or 2

    if not os.path.exists(directory + fname):
        print("\nWARNING: file " + fname + " does not exist")
        return [], [], 0, 0

    f = open(directory + fname, "r")
    times0 = []
    times1 = []
    sizes0 = []
    sizes1 = []

    # By default, args.num_consecutive_packets_in and args.num_consecutive_packets_out = 20
    avg_consecutive_packets = {-1: num_consecutive_packets_in, 1:num_consecutive_packets_out}
    # note: ceil of exp is geometric (args.num_conspack_dist = exp by default, which executes random.expovariate(float(1)/20))
    consecutive_packets = {-1: int(math.ceil(get_exp(avg_consecutive_packets[-1], num_conspack_dist))),
                           1: int(math.ceil(get_exp(avg_consecutive_packets[1], num_conspack_dist)))}
    current_cons_packets = {-1: consecutive_packets[-1], 1: consecutive_packets[1]}
   
    current_is_sec = {-1: False, 1: False}
    p_loc = {-1: p_in, 1: p_out}

    if change_times > 0:
        p_loc = {-1: get_local_p(p_in, p_dist), 1: get_local_p(p_out, p_dist)}
        next_stop = get_exp(change_times, change_dist)
    elif change_packets > 0:
        p_loc = {-1: get_local_p(p_in, p_dist), 1: get_local_p(p_out, p_dist)}
        change_packets = get_exp(change_packets, change_dist)
    else: p_loc = {-1: p_in, 1: p_out}

    size_before = 0
    num_packets_sent = 0

    ### change it to handle the array instead
    xs = f.readlines()
    for x in xs: # line by line in the file.
        num_packets_sent += 1
        ts = float(x.split('\t')[0]) # timestamp
        size = int(x.split('\t')[1]) # size
        size_before += size

        if change_times > 0:
            assert(ts != -1)  # if length is 1, no timestamp is given
            if ts >= next_stop:
                p_loc = {-1: get_local_p(p_in, p_dist), 1: get_local_p(p_out, p_dist)}
                next_stop += get_exp(change_times, change_dist)
        elif change_packets > 0:
            if num_packets_sent == change_packets:
                p_loc = {-1: get_local_p(p_in, p_dist), 1: get_local_p(p_out, p_dist)}
                change_packets = int(math.ceil(get_exp(change_packets, change_dist)))
                num_packets_sent = 0
        size_before += size
        # select packets to send on the "secure" channel (they are simply removed)
        # must send consecutive_packets on same channel
        if current_cons_packets[np.sign(size)] < consecutive_packets[np.sign(size)]:
            current_cons_packets[np.sign(size)] += 1
            send_on_sec = current_is_sec[np.sign(size)]
        else: # so sending c packets is done, then pick another c.
            consecutive_packets[np.sign(size)] = int(math.ceil(get_exp(avg_consecutive_packets[np.sign(size)], num_conspack_dist)))
            # initialize current packet count to 1
            current_cons_packets[np.sign(size)] = 1
            if random.random() <= p_loc[np.sign(size)]: # Network0
                send_on_sec = False
            else: # Network1
                send_on_sec = True
        current_is_sec[np.sign(size)] = send_on_sec
                
        if send_on_sec:
            times1.append(ts)
            sizes1.append(size)
            selected_network.append(1)
        else:
            times0.append(ts)
            sizes0.append(size)
            selected_network.append(0)
    f.close()

    num_features = 5000
    if len(sizes0+sizes1) == 0:
        return num_features * [0], num_features * [0], 0

    feature_dict = {
    'TikTok': getTikTok,
    'Direction': getDirection,
    '1-DTAM': get1_DTAM,
    'ICD': getICD,
    'ICDS': getICDS,
    }

    features0, features1 = feature_dict.get(FEATURE)(sizes0, sizes1, times0, times1)
    features0 = np.pad (features0[:num_features], (0, num_features - len (features0[:num_features])), 'constant', constant_values=0)
    features1 = np.pad (features1[:num_features], (0, num_features - len (features1[:num_features])), 'constant', constant_values=0)
    return np.array(features0), np.array(features1), selected_network


def get_local_p(p, p_dist):
    if p == 0 or p == 1:
        loc_p = p
    else:
        if p_dist == "uniform":
            s = round(min(p, 1-p),2)
            loc_p = p - s + 2*s*np.random.uniform()
        elif p_dist == "fixed":
            # fixed
            loc_p = p
        elif p_dist == "custom":
            loc_p = np.random.uniform(p-0.1,p,1)
            return loc_p
    return min(1,max(0,loc_p))  # between 0 and 1

def fextractor(p_train, FEATURE):
    t_beg = time.time()

    original_directory = "/DeepCoAST/HyWF_raw/" # this folder contains the data before splitting
    directory_train = os.path.join(original_directory, "train/")
    directory_test = os.path.join(original_directory, "test/")
    test_files = listdir(directory_test)
    train_files = listdir(directory_train)

    print('train_files count',len(train_files))
    print('test_files count', len (test_files))

    # test files
    X_test0 = []
    X_test1 = []
    y_test0 = []
    y_test1 = []
    total_num_pkts_network0_test = 0
    total_num_pkts_network1_test = 0
    total_num_pkts_test = 0

    # set param by DeepCoFFEA (Oh et al.)
    p_test_in = 0.5
    p_test_out = 0.5
    p_dist = "uniform"

    for fname in test_files:
        p_test_in_loc = get_local_p(p_test_in, p_dist)
        p_test_out_loc = get_local_p(p_test_out, p_dist)
        original_probas = {-1: p_test_in, 1: p_test_out}

        try:
            features_loc0, features_loc1, selected_network = write_features(directory_test, fname, p_test_in_loc, p_test_out_loc, p_dist, False, original_probas,FEATURE)
        except ValueError as e: pass

        total_num_pkts_network0_test += selected_network.count(0)
        total_num_pkts_network1_test += selected_network.count(1)
        X_test0.append(features_loc0)
        X_test1.append(features_loc1)
        y_test0.append(int(fname.split('-')[0]))
        y_test1.append(int(fname.split('-')[0]))

    total_num_pkts_test = total_num_pkts_network0_test + total_num_pkts_network1_test
    sys.stdout.write("\nThe ratio of selections between Network0 and Network1 calculated based on # of packets: %f:%f" % (total_num_pkts_network0_test*100/total_num_pkts_test, total_num_pkts_network1_test*100/total_num_pkts_test))
    sys.stdout.write("\nTest files done\n")
    sys.stdout.flush()

    # Train files
    X_train0 = []
    X_train1 = []
    y_train0 = []
    y_train1 = []
    total_num_pkts_network0_train = 0
    total_num_pkts_network1_train = 0
    total_num_pkts_train = 0

    count_exp = 0
    # param by DeepCoFFEA (Oh et al.)
    p_train_in = p_train
    p_train_out = p_train
    p_dist = "uniform"

    for fname in tqdm(train_files):
        p_train_in_loc = get_local_p (p_train_in, p_dist)
        p_train_out_loc = get_local_p (p_train_out, p_dist)

        original_probas = {-1: p_train_in, 1: p_train_out}

        try:
            features_loc0, features_loc1, selected_network = write_features\
            (directory_train, fname, p_train_in_loc, p_train_out_loc, p_dist, True, original_probas, FEATURE)
        except ValueError as e: pass
        
        total_num_pkts_network0_train += selected_network.count(0)
        total_num_pkts_network1_train += selected_network.count(1)
        X_train0.append(features_loc0)
        X_train1.append(features_loc1)
        y_train0.append(int(fname.split('-')[0]))
        y_train1.append(int(fname.split('-')[0]))

    total_num_pkts_train = total_num_pkts_network0_train + total_num_pkts_network1_train
    sys.stdout.write("\nThe ratio of selections between Network0 and Network1 calculated based on # of packets: %f:%f" % (total_num_pkts_network0_train*100/total_num_pkts_train, total_num_pkts_network1_train*100/total_num_pkts_train))
    sys.stdout.write("\nTrain files done\n")
    sys.stdout.flush()

    count_exp += 1

    sys.stdout.write("\nDone in "+str(time.time()-t_beg)+" seconds\n")
    sys.stdout.flush()
    return X_train0, y_train0, X_train1, y_train1, X_test0, y_test0, X_test1, y_test1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", '--feature', type=str, default = "1-DTAM")
    args = parser.parse_args()
    FEATURE = args.feature
    
    X_train0, y_train0, X_train1, y_train1, X_test0, y_test0, X_test1, y_test1 = fextractor(0.5, FEATURE)
    
    print ('Shape of X_train0:', np.array(X_train0).shape)
    print ('Shape of y_train0', np.array(y_train0).shape)
    print ('Shape of X_test0', np.array(X_test0).shape)
    print ('Shape of y_test0', np.array(y_test0).shape)

    print ('Shape of X_train1', np.array(X_train1).shape)
    print ('Shape of y_train1', np.array(y_train1).shape)
    print ('Shape of X_test1', np.array(X_test1).shape)
    print ('Shape of y_test1', np.array(y_test1).shape)

    if FEATURE == "1-DTAM": output_path = "/DeepCoAST/HyWF/1-DTAM/" 
    else: output_path = "/DeepCoAST/HyWF/" + FEATURE + "/"
    if not os.path.isdir(output_path): os.mkdir(output_path)

    with open(output_path+"train_path0.pkl", 'wb') as f:
        pickle.dump(np.array(X_train0), f, pickle.HIGHEST_PROTOCOL)
    with open(output_path+"test_path0.pkl", 'wb') as f:
        pickle.dump(np.array(X_test0), f, pickle.HIGHEST_PROTOCOL)
    with open(output_path+"train_path1.pkl", 'wb') as f:
        pickle.dump(np.array(X_train1), f, pickle.HIGHEST_PROTOCOL)
    with open(output_path+"test_path1.pkl", 'wb') as f:
        pickle.dump(np.array(X_test1), f, pickle.HIGHEST_PROTOCOL)
    with open(output_path+"train_label.pkl", 'wb') as f:
        pickle.dump(np.array(y_train0), f, pickle.HIGHEST_PROTOCOL)
    with open(output_path+"test_label.pkl", 'wb') as f:
        pickle.dump(np.array(y_test0), f, pickle.HIGHEST_PROTOCOL)

    ## check data     
    # data = np.load(output_path+"train_path1.pkl", allow_pickle=True)
    # print(data[0][:50])
    # print(data[22][:50])
    # data = np.load(output_path+"train_label.pkl", allow_pickle=True)
    # print(data[0])
    # print(data[22])
    # print()
    # data = np.load(output_path+"test_path0.pkl", allow_pickle=True)
    # print(data[0][:50])
    # print(data[22][:50])
    # data = np.load(output_path+"test_label.pkl", allow_pickle=True)
    # print(data[0])
    # print(data[22])

    print("All Done")

    
