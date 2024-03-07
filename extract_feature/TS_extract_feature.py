import numpy as np
from natsort import natsorted
from tqdm import tqdm
import glob
import pickle
from datetime import datetime
import argparse
from utility import *
import os

parser = argparse.ArgumentParser()
parser.add_argument("-r", '--range_', type=str, default = "10_40") # 10_40 or 50_70
parser.add_argument("-p", '--prob', type=str, default = "0.5_0.5") # 0.3_0.7, 0.4_0.6, 0.5_0.5
parser.add_argument("-f", '--feature', type=str, default = "1-DTAM")

INPUT_SIZE = 5000
SITE = 95

feature_dict = {
    'TikTok': getTikTok,
    'Direction': getDirection,
    '1-DTAM': get1_DTAM,
    'ICD': getICD,
    'ICDS': getICDS,
    }


def getFeature(feature, instance):
    feature_func = feature_dict.get(feature)
    return feature_func(instance)

def getMetadata(file_name):
    site, instance = map(int, file_name.split('/')[-1].split('.')[0].split('_')[0].split('-'))
    return site, instance

def extract(input_path0, input_path1, feature):
    files0 = natsorted(glob.glob(input_path0+'/*split*.cell'))
    files1 = natsorted(glob.glob(input_path1+'/*split*.cell'))
    test_check = [False] * SITE

    train_path0 = []
    train_path1 = []
    train_label = []
    test_path0 = []
    test_path1 = []
    test_label = []
    
    print("Number of Samples Before Deleting Blank Files:", len(files0) + len(files1))

    for f0, f1 in tqdm(zip(files0, files1), total=len(files1)):
        with open(f0, 'r') as fo0:
            lines = fo0.readlines()
            if len(lines) < 10: continue
            path0_data = getFeature(feature, lines)
        with open(f1, 'r') as fo1:
            lines = fo1.readlines()
            if len(lines) < 10: continue
            path1_data = getFeature(feature, lines)
        path0_data = np.concatenate((path0_data, np.zeros(max(0, INPUT_SIZE - len(path0_data)))))[:INPUT_SIZE]
        path1_data = np.concatenate((path1_data, np.zeros(max(0, INPUT_SIZE - len(path1_data)))))[:INPUT_SIZE]

        site, instance = getMetadata(f1)
        if not test_check[site]:
            test_path0.append(path0_data)
            test_path1.append(path1_data)
            test_label.append(site)
            test_check[site] = True
        else:
            train_path0.append(path0_data)
            train_path1.append(path1_data)
            train_label.append(site)

    print("Number of Samples After Deleting Blank Files:", (len(train_label)+len(test_label))*2)
    print("Number of Train Set:", len(train_path0)+len(train_path1))
    print("Number of Test Set:", len(test_path0)+len(test_path1))
    return train_path0, train_path1, test_path0, test_path1, train_label, test_label

if __name__ == "__main__":
    args = parser.parse_args()
    r = args.range_
    p = args.prob
    feature = args.feature
    input_path = "/DeepCoAST/TS_split2_raw/" + r +"/" + p + "/"
    input_path0 = input_path + "path0/" 
    input_path1 = input_path + "path1/"

    train_path0, train_path1, test_path0, test_path1, train_label, test_label = extract(input_path0, input_path1, feature=feature)
    print(train_path1[5][30:])
    print(train_path0[5][0:])
    print(train_path0[5][15:])
    print(train_path0[5][30:])
    print(train_path0[5][100:])
    print(train_path1[10][30:])
    output_path = "/DeepCoAST/TrafficSliver/n(" + r +")/w(" + p + ")/" +feature
    if not os.path.isdir(output_path): os.mkdir(output_path)

    with open(output_path + '/train_path0.pkl', 'wb') as f:
            pickle.dump(train_path0, f, pickle.HIGHEST_PROTOCOL)
    with open(output_path + '/train_path1.pkl', 'wb') as f:
            pickle.dump(train_path1, f, pickle.HIGHEST_PROTOCOL)
    with open(output_path + '/test_path0.pkl', 'wb') as f:
            pickle.dump(test_path0, f, pickle.HIGHEST_PROTOCOL)
    with open(output_path + '/test_path1.pkl', 'wb') as f:
            pickle.dump(test_path1, f, pickle.HIGHEST_PROTOCOL)
    with open(output_path + '/train_label.pkl', 'wb') as f:
            pickle.dump(train_label, f, pickle.HIGHEST_PROTOCOL)
    with open(output_path + '/test_label.pkl', 'wb') as f:
            pickle.dump(test_label, f, pickle.HIGHEST_PROTOCOL)
    print("All Done")
