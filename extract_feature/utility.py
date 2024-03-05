#last modified: 24/3/5
import numpy as np

def getData(line):
    timestamp, dir, size = map(float, line.split('\t'))
    dir = int(dir)
    return timestamp, dir, size

def getDirection(instance):
    dirs = []
    for line in map(str.strip, instance):
        _, dir, _ = getData(line)
        dirs.append(dir)
    return dirs

def getTikTok(instance):
    feature = []
    for line in map(str.strip, instance):
        timestamp, dir, _ = getData(line)
        feature.append(timestamp * dir)
    return feature

def get1_DTAM(instance):
    max_matrix_len = 1800
    maximum_load_time = 80
    timestamps = []
    dirs = []
    for line in map(str.strip, instance):
        timestamp, dir, _ = getData(line)
        timestamps.append(timestamp)
        dirs.append(dir)
    if timestamps:
        feature = [[0 for _ in range(max_matrix_len)], [0 for _ in range(max_matrix_len)]]
        for i in range(0, len(dirs)):
            if dirs[i] > 0:
                if timestamps[i] >= maximum_load_time:
                    feature[0][-1] += 1
                else:
                    idx = int(timestamps[i] * (max_matrix_len - 1) / maximum_load_time)
                    feature[0][idx] += 1
            if dirs[i] < 0:
                if timestamps[i] >= maximum_load_time:
                    feature[1][-1] += 1
                else:
                    idx = int(timestamps[i] * (max_matrix_len - 1) / maximum_load_time)
                    feature[1][idx] += 1
        feature1 = feature[0]
        feature2 = feature[1]
        result = feature1 + feature2
    return result

def getICD(instance):
    timestamps = []
    dirs = []
    first = True
    for line in map(str.strip, instance):
        timestamp, dir, _ = getData(line)
        if first == True:
            first = False
            timestamps.append(0)
        else:
            timestamps.append(timestamp-beforetimestamp)
        dirs.append(dir)
        beforetimestamp = timestamp
    return timestamps

def getICDS(instance):
    timestamps = []
    dirs = []
    first = True
    for line in map(str.strip, instance):
        timestamp, dir, _ = getData(line)
        if first == True:
            first = False
            timestamps.append(0)
        else:
            timestamps.append(timestamp-beforetimestamp)
        dirs.append(dir)
        beforetimestamp = timestamp
    return timestamps+[512 * x for x in dirs]


