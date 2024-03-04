
from __future__ import print_function
from keras.callbacks import LambdaCallback
from new_model import create_model, create_model_2d
import tensorflow as tf
import os
from keras.models import Model
from keras.layers import Input, Lambda, Dot
from tensorflow.keras.optimizers import Adam
import sys
import numpy as np
import argparse
import gc
import pickle
from keras import backend as K
import argparse

import time
import datetime

# for using GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
parser = argparse.ArgumentParser()

def get_params():
    parser.add_argument("-r", '--range_', type=str, default = "50,70") # parameter for TrafficSliver range(n), (ex: 10,40 , 50,70)
    parser.add_argument("-p", '--prob', type=str, default = "0.5, 0.5") # parameter for TrafficSliver weight(w), (ex: 0.3, 0.7 , 0.4, 0.6 , 0.5, 0.5)
    parser.add_argument("-a", '--alpha', type=float, default = 0.1)
    parser.add_argument("-b", '--batch_size', type=int, default = 128)
    parser.add_argument("-f", '--feature', type=str, default = "Direction") # Direction, TikTok, 1-D TAM, ICD, ICDS
    parser.add_argument("-e", '--epoch', type=int, default = 300) # if you use max size epoch, enter -1 
    parser.add_argument("-i", '--input_size', type=int, default = 5000) 
    parser.add_argument("-d", '--defense_type', type=str, default = "TrafficSliver") # TrafficSliver, CoMPS, HyWF
    parser.add_argument("-s", '--setting', type=int, default = 150) 
    args = parser.parse_args()
    return args

def get_session(gpu_fraction=0.85):
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)
    return tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# customized loss
def cosine_triplet_loss(X):
    _alpha = alpha_value
    positive_sim, negative_sim = X

    losses = K.maximum(0.0, negative_sim - positive_sim + _alpha)
    return K.mean(losses)

def identity_loss(y_true, y_pred):
        return K.mean(y_pred - 0 * y_true)

def intersect(a, b):
        return list(set(a) & set(b))

def build_similarities(conv1, tor_t, exit_t):

        tor_embs = conv1.predict(tor_t)
        exit_embs = conv1.predict(exit_t)
        all_embs = np.concatenate((tor_embs, exit_embs), axis=0)
        all_embs = all_embs / np.linalg.norm(all_embs, axis=-1, keepdims=True)
        mid = int(len(all_embs) / 2)
        all_sims = np.dot(all_embs[:mid], all_embs[mid:].T)
        return all_sims

def build_negatives(anc_idxs, pos_idxs, similarities, neg_imgs_idx, num_retries=50):
        # If no similarities were computed, return a random negative
        if similarities is None:
            anc_idxs = list(anc_idxs)
            valid_neg_pool = neg_imgs_idx
            print('valid_neg_pool', valid_neg_pool.shape)
            return np.random.choice(valid_neg_pool, len(anc_idxs), replace=False)
        final_neg = []
        # for each positive pair
        for (anc_idx, pos_idx) in zip(anc_idxs, pos_idxs):
            anchor_class = anc_idx
            valid_neg_pool = neg_imgs_idx 
            # positive similarity
            sim = similarities[anc_idx, pos_idx]
            # find all negatives which are semi(hard)
            possible_ids = np.where((similarities[anc_idx] + alpha_value) > sim)[0]
            possible_ids = intersect(valid_neg_pool, possible_ids)
            appended = False
            for iteration in range(num_retries):
                if len(possible_ids) == 0:
                    break
                idx_neg = np.random.choice(possible_ids, 1)[0]
                if idx_neg != anchor_class:
                    final_neg.append(idx_neg)
                    appended = True
                    break
            if not appended:
                final_neg.append(np.random.choice(valid_neg_pool, 1)[0])
        return final_neg

class SemiHardTripletGenerator():
        def __init__(self, Xa_train, Xp_train, batch_size, neg_traces_train_idx, Xa_train_all, Xp_train_all, conv1):
            self.batch_size = batch_size  # 128

            self.Xa = Xa_train
            self.Xp = Xp_train
            self.Xa_all = Xa_train_all
            self.Xp_all = Xp_train_all
            self.Xp = Xp_train
            self.cur_train_index = 0
            self.num_samples = Xa_train.shape[0]
            self.neg_traces_idx = neg_traces_train_idx

            if conv1:
                self.similarities = build_similarities(conv1, self.Xa_all,
                                                       self.Xp_all)  # compute all similarities including cross pairs
            else:
                self.similarities = None

        def next_train(self):
            while 1:
                self.cur_train_index += self.batch_size
                if self.cur_train_index >= self.num_samples:
                    self.cur_train_index = 0  # initialize the index for the next epoch
                # fill one batch
                traces_a = np.array(range(self.cur_train_index,
                                          self.cur_train_index + self.batch_size))
                traces_p = np.array(range(self.cur_train_index,
                                          self.cur_train_index + self.batch_size))

                traces_n = build_negatives(traces_a, traces_p, self.similarities, self.neg_traces_idx)
                
                try:
                    yield ([self.Xa[traces_a],
                            self.Xp[traces_p],
                            self.Xp_all[traces_n]],
                        np.zeros(shape=(traces_a.shape[0]))
                        )
                except:
                    traces_a = np.array(range(len(traces_a)-self.batch_size,
                                          len(traces_a)))
                    traces_p = np.array(range(len(traces_p)-self.batch_size,
                                          len(traces_p)))

                    traces_n = build_negatives(traces_a, traces_p, self.similarities, self.neg_traces_idx)
                    yield ([self.Xa[traces_a],
                            self.Xp[traces_p],
                            self.Xp_all[traces_n]],
                        np.zeros(shape=(traces_a.shape[0]))
                        )

def saveModel(epoch, logs):
    global best_loss

    loss = logs['loss']

    if loss < best_loss:
        print("loss is improved from {} to {}. save the model".format(str(best_loss),
                                                                        str(loss)))
        best_loss = loss
        print(output_path)
        shared_model.save_weights(output_path+ "/epoch_{}_loss_{}_model.h5".format(str(current_epoch), str(loss)))
    else:
        print("loss is not improved from {}.".format(str(best_loss)))


if __name__ == '__main__':
    start = time.time()

    K.set_session(get_session())
    args = get_params()
    alpha_value = float(args.alpha)
    feature = args.feature
    r = args.range_
    p = args.prob
    batch_size = args.batch_size
    INPUT_SIZE = args.input_size
    input_shape = (INPUT_SIZE, 1)
    
    if args.epoch == -1: MYEPOCH = sys.maxsize
    else: MYEPOCH = args.epoch
    defense_type = args.defense_type
    if defense_type == "HyWF":
        input_path = "/DeepCoAST/"+defense_type + "/"+feature+"/"
    elif defense_type == "CoMPS":
         input_path = "/DeepCoAST/"+defense_type + "/"+feature+"/"
    else:
         input_path = "/DeepCoAST/TrafficSliver/n(" + r +")/w(" + p + ")/" +feature+"/"

    with open(input_path + 'train_path1.pkl', 'rb') as f:
        train_path1=pickle.load(f)
    with open(input_path + 'train_path0.pkl', 'rb') as f:
        train_path0=pickle.load(f)
    with open(input_path + 'test_path0.pkl', 'rb') as f:
        test_path0=pickle.load(f)
    with open(input_path + 'test_path1.pkl', 'rb') as f:
        test_path1=pickle.load(f)
    with open(input_path + 'train_label.pkl', 'rb') as f:
        train_label=pickle.load(f)

    train_path1 = np.array(train_path1)
    train_path0 = np.array(train_path0)
    test_path1 = np.array(test_path1)
    test_path0 = np.array(test_path0)

    # move to output foler of h5
    if defense_type == "HyWF" or defense_type == "CoMPS":
        if MYEPOCH == sys.maxsize: 
            if feature == 'ICD1000' or feature == 'ICDS1000':
                output_path = "/DeepCoAST/"+ defense_type + "/" + feature + "/b"+ str(batch_size)+"_"+feature+"_maxsize"
            else: output_path = "/DeepCoAST/"+ defense_type + "/" + feature + "/b"+ str(batch_size)+"_"+feature[:3]+"_maxsize"
        else:
            if feature == 'ICD1000' or feature == 'ICDS1000':
                output_path = "/DeepCoAST/"+ defense_type + "/" + feature + "/b"+ str(batch_size)+"_"+feature+"_"+str(MYEPOCH)
            else: output_path = "/DeepCoAST/"+ defense_type + "/" + feature + "/b"+ str(batch_size)+"_"+feature[:3]+"_"+str(MYEPOCH)
             
    else: 
        if MYEPOCH == sys.maxsize: 
            if feature == 'ICD1000' or feature == 'ICD1000' or feature == 'ICDS100' or feature == 'ICD100' or feature == 'ICDS10' or feature == 'ICD10':
                output_path = "/DeepCoAST/"+ defense_type + "/" + r +"/" + p + "/" +feature+"/b"+ str(batch_size)+"_"+feature+"_maxsize"
            else: output_path = "/DeepCoAST/"+ defense_type + "/" + r +"/" + p + "/" +feature+"/b"+ str(batch_size)+"_"+feature[:3]+"_maxsize"
        else:
            if feature == 'ICD1000' or feature == 'ICD1000' or feature == 'ICDS100' or feature == 'ICD100' or feature == 'ICDS10' or feature == 'ICD10':
                output_path = "/DeepCoAST/"+ defense_type + "/" + r +"/" + p + "/" +feature+"/b"+ str(batch_size)+"_"+feature+"_"+str(MYEPOCH)
            else: output_path = "/DeepCoAST/"+ defense_type + "/" + r +"/" + p + "/" +feature+"/b"+ str(batch_size)+"_"+feature[:3]+"_"+str(MYEPOCH)
    if not os.path.isdir(output_path):os.mkdir(output_path)
    shared_model = create_model(input_shape=input_shape, emb_size=64, model_name='all-in-one')

    anchor = Input(input_shape, name='anchor')
    positive = Input(input_shape, name='positive')
    negative = Input(input_shape, name='negative')

    a = shared_model(anchor)
    p = shared_model(positive)
    n = shared_model(negative)

    print('a shape', a.shape)
    print('p shape', p.shape)
    print('n shape', n.shape)
    pos_sim = Dot(axes=-1, normalize=True)([a, p])
    neg_sim = Dot(axes=-1, normalize=True)([a, n])
    print('pos_sim shape', pos_sim.shape)
    print('neg_sim shape', neg_sim.shape)

    loss = Lambda(cosine_triplet_loss, output_shape=(1,))([pos_sim, neg_sim])

    model_triplet = Model(
        inputs=[anchor, positive, negative],
        outputs=loss)
    print(model_triplet.summary())

    opt = Adam(learning_rate=0.001, decay=1e-6)
    model_triplet.compile(loss=identity_loss, optimizer=opt)

    # At first epoch we don't generate hard triplets
    all_traces_train_idx = np.array(range(len(train_label)))
    gen_hard = SemiHardTripletGenerator(train_path0, train_path1, batch_size, all_traces_train_idx,
                                        train_path0, train_path1, None)

    best_loss = sys.float_info.max
    global current_epoch
    current_epoch = 0
    while (current_epoch < MYEPOCH):
        print("built new hard generator for epoch " + str(current_epoch))

        if current_epoch % 2 == 0:
            if current_epoch == 0:
                model_triplet.fit_generator(generator=gen_hard.next_train(),
                                            steps_per_epoch=train_path0.shape[0] // batch_size - 1,
                                            epochs=1, verbose=2)
            else:
                model_triplet.fit_generator(generator=gen_hard_even.next_train(),
                                            steps_per_epoch=(train_path0.shape[0] // 2) // batch_size - 1,
                                            epochs=1, verbose=2, callbacks=[LambdaCallback(on_epoch_end=saveModel)])
        else:
            model_triplet.fit_generator(generator=gen_hard_odd.next_train(),
                                        steps_per_epoch=(train_path0.shape[0] // 2) // batch_size - 1,
                                        epochs=1, verbose=2, callbacks=[LambdaCallback(on_epoch_end=saveModel)])
        gc.collect()
        K.clear_session()
        current_epoch += 1
        mid = int(len(train_path0) / 2)
        random_ind = np.array(range(len(train_path0)))
        np.random.shuffle(random_ind)
        X1 = np.array(random_ind[:mid])
        X2 = np.array(random_ind[mid:])

        gen_hard_odd = SemiHardTripletGenerator(train_path0[X1], train_path1[X1], batch_size, X2, train_path0,
                                                train_path1,
                                                shared_model)
        gen_hard_even = SemiHardTripletGenerator(train_path0[X2], train_path1[X2], batch_size,
                                                 X1, train_path0, train_path1,
                                                 shared_model)
    end = time.time()
    sec = (end - start)
    result_list = str(datetime.timedelta(seconds=sec)).split(".")

    print("All done")
    print(f"h5 folder path: {output_path}")
    print(f"{MYEPOCH=} \n {INPUT_SIZE=} \n{batch_size=} \n{feature=}\n{defense_type=}\n")
    print(result_list[0])
