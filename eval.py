import tensorflow as tf
from new_model import create_model
import numpy as np
import os
from keras import backend as K
from sklearn.metrics.pairwise import cosine_similarity
import csv
import time
import argparse
import pickle

total_emb = 0
total_vot = 0
total_cos = 0

# for using GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
parser = argparse.ArgumentParser()

def get_params():
    parser.add_argument("-a", '--alpha', type=float, default = 0.1)
    parser.add_argument("-f", '--feature', type=str, default = "Direction") # Direction, 1-DTAM, ICD, ICDS, TikTok
    parser.add_argument("-d", '--defense', type=str, default = "TrafficSliver") # TrafficSliver, HyWF, CoMPS
    parser.add_argument("-s", '--setting', type=int, default = 300)
    parser.add_argument ("-m", '--model_path') # model_path
    args = parser.parse_args()
    return args


def get_session(gpu_fraction=0.85):
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)
    return tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


def ini_cosine_output(single_output_l, input_number):
    for pairs in range (0, (input_number * input_number)):
        single_output_l.append (0)

def Cosine_Similarity_eval(path0_embs, path1_embs, similarity_threshold, single_output_l, cosine_similarity_all_list, muti_output_list):
    global total_vot
    number_of_lines = path0_embs.shape[0]
    start_emd = time.time ()
    for path0_emb_index in range (0, number_of_lines):
        t = similarity_threshold[path0_emb_index]
        constant_num = int (path0_emb_index * number_of_lines)
        for path1_emb_index in range (0, number_of_lines):
            if (cosine_similarity_all_list[path0_emb_index][path1_emb_index] >= t):
                single_output_l[constant_num + path1_emb_index] = single_output_l[constant_num + path1_emb_index] + 1

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    
    for path0_eval_index in range (0, path0_embs.shape[0]):
        for path1_eval_index in range (0, path0_embs.shape[0]):
            cos_condithon_a = (path0_eval_index == path1_eval_index)
            cos_condition_c = False
            cos_condition_b = False
            
            if single_output_l[(path0_eval_index * (path0_embs.shape[0])) + path1_eval_index] > 0:
                cos_condition_b = True
            else: cos_condition_c = True

            if (cos_condithon_a and cos_condition_b):
                TP = TP + 1
            if (cos_condithon_a and cos_condition_c):
                FN = FN + 1
            if ((not (cos_condithon_a)) and cos_condition_b):
                FP = FP + 1
            if ((not (cos_condithon_a)) and cos_condition_c):
                TN = TN + 1

    if ((TP + FN) != 0):
        TPR = (float) (TP) / (TP + FN)
    else:
        TPR = -1

    if ((FP + TN) != 0):
        FPR = (float) (FP) / (FP + TN)
    else:
        FPR = -1

    muti_output_list.append (TPR)
    muti_output_list.append (FPR)
    muti_output_list.append (calculate_bdr (TPR, FPR))
    print(TPR,FPR,calculate_bdr (TPR, FPR))

    end_time = time.time ()
    total_vot = total_vot + (end_time - start_emd)

def calculate_bdr(tpr, fpr):
   
    TPR = tpr
    FPR = fpr
    c = 1 / flow
    u = (flow-1) / flow
    if ((TPR * c) + (FPR * u)) != 0:
        BDR = (TPR * c) / ((TPR * c) + (FPR * u))
    else:
        BDR = -1
    return BDR


# Every tor flow will have a unique threshold
def threshold_finder(input_similarity_list, gen_ranks, thres_seed, use_global): # get top 60 probabilities
    output_shreshold_list = []
    for simi_list_index in range (0, len (input_similarity_list)):
        temp = list (input_similarity_list[simi_list_index])
        temp.sort (reverse=True)

        cut_point = int ((len (input_similarity_list[simi_list_index]) - 1) * ((thres_seed) / 100))
        if use_global == 1:
            output_shreshold_list.append (thres_seed) 
        elif use_global != 1:
            output_shreshold_list.append (temp[cut_point])
    return output_shreshold_list

def eval_model(full_or_half, use_new_data, model_path, test_path0, test_path1, thr, use_global,
               muti_output_list, soft_muti_output_list):
    global total_emb
    global total_vot
    global total_cos
    INPUT_SIZE = 5000

    test_path0 = np.array(test_path0)
    test_path1 = np.array(test_path1)

    print('test_path0 sample 개수: ', np.array(test_path0).shape)
    print('test_path1 sample 개수: ', np.array(test_path1).shape)
    print('flow:', full_or_half)

    pad_path0 = INPUT_SIZE

    model = create_model (input_shape=(pad_path0, 1), emb_size=64, model_name='all-in-one')
    
    # load triplet models for tor and exit traffic
    model.load_weights (model_path)
    
    # This list will be the output list of cosine similarity approach.
    single_output = []
    cosine_similarity_table = []
    threshold_result = []

    # below are the code that are used for controlling the behavior of the program
    thres_seed = thr
    
    start_emd = time.time ()
    path0_embs = model.predict (test_path0)
    path1_embs = model.predict (test_path1)
    end_emd = time.time ()
    # print('[#####] Time for embedding: ', end_emd - start_emd, 'sec')
    total_emb = total_emb + (end_emd - start_emd)
    print("path0_embs shape:", path0_embs.shape)
    print("path1_embs shape:", path1_embs.shape)
    print("init the final cosine similarity output now.....")
    ini_cosine_output (single_output, path0_embs.shape[0])
    # print("getting cosine similarity results......")
    start_cos = time.time ()
    cosine_similarity_table = cosine_similarity (path0_embs, path1_embs) # 여기까지 괜찮음
    end_cos = time.time ()
    # print('[#####] Time for cosine: ', end_cos - start_cos, 'sec')
    total_cos = total_cos + (end_cos - start_cos)
    threshold_result = threshold_finder (cosine_similarity_table, 0, thres_seed, use_global)
    print("single output:", np.array(single_output).shape)
    Cosine_Similarity_eval (path0_embs, path1_embs, threshold_result, single_output, cosine_similarity_table, muti_output_list)


if __name__ == "__main__":
    # if you don't use GPU, comment out the following
    K.set_session(get_session())
    args = get_params()
    feature = args.feature
    defense = args.defense
    model_path = args.model_path
    start_time = time.time ()

    model_path_list = model_path.split('/')
    print(model_path_list)
    input_path = '/'.join(model_path_list[:-2])
    print(input_path)
    with open('/DeepCoAST/'+defense+'/'+feature + '/test_path0.pkl', 'rb') as f:
        test_path0=pickle.load(f)
    with open('/DeepCoAST/'+defense+'/'+feature + '/test_path1.pkl', 'rb') as f:
        test_path1=pickle.load(f)
    output_path = '/'.join(model_path.split('/')[:-1]) + '/'
    print(output_path)
    
    # change the threshold list appropriately
    # rank_thr_list = [60, 50,47,43,40,37,33,28,24,20,16.667,14,12.5,11,10,9,8.333,7,6.25,5,4.545,3.846,2.941,1.667,1.6,1.5,1.4,1.3,1.2,1.1,1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2]#[10]
    rank_thr_list = [100, 90, 80, 70, 60, 50,47,43,40,37,33,28,24,20,16.667,14,12.5,11,10,9,8.333,7,6.25,5,4.545,3.846,2.941,1.667,1.6,1.5,1.4,1.3,1.2,1.1,1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2, 0]#[10]
    num_of_thr = len(rank_thr_list)

    flow_length = np.array(test_path0).shape[0]
    global flow
    flow = flow_length
    print(f"{flow_length=}")

    rank_multi_output = []
    five_rank_multi_output = []
    for i in range (0, num_of_thr):
        rank_multi_output.append ([(rank_thr_list[i])])
        five_rank_multi_output.append ([(rank_thr_list[i])])

    epoch_index = 0
    use_global = 0
    use_new_data = 0

    for thr in rank_thr_list:
        eval_model (flow_length, use_new_data, model_path, test_path0, test_path1, thr, use_global,
                    rank_multi_output[epoch_index], [])
        epoch_index = epoch_index + 1
    end_time = time.time ()
    with open(output_path + model_path.split('/')[-1].split('.h5')[0]+'.csv', "w", newline="") as rank_f:
        writer = csv.writer(rank_f)
        writer.writerows(rank_multi_output)

    print (f"total: {str (end_time-start_time)}sec")



