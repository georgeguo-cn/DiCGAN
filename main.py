"""
    @Time   : 2020.01.16
    @Author : Zhiqiang Guo
    @Email  : zhiqiangguo@hust.edu.cn
    This file is the run file. It contains some arguments and train processing.
"""

import argparse
import tensorflow as tf
import numpy as np
import random
from tqdm import tqdm
from time import time

import parameters
from data import Data
from DiCGAN import DiCGAN
from evaluation import evaluate_ranking

tf.set_random_seed(20200115)

def to_one_hot(y, n_class):
     return np.eye(n_class)[y]

if __name__ == '__main__':
    filename = 'Ciao'
    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument('--dataname', type=str, default=filename)
    parser.add_argument('--data_root', type=str, default='datasets/'+ filename + '/')
    parser.add_argument('--L', type= int, default=100)
    parser.add_argument('--percentage', type=float, default=0.8)
    parser.add_argument('--emb_len', type=int, default=64)
    parser.add_argument('--kernel_size', type=int, default=[3,3,5,6,5,4,3])   # L=10, k=[2,3,3], d=[1,2,1]  #L=50, k=[3,3,4,4,3], d=[1,2,4,5,3], L=100, k=[3,3,5,6,5,4,3],d=[1,2,4,6,4,3,1]
    parser.add_argument('--dilations', type=list, default=[1,2,4,6,4,3,1])
    parser.add_argument('--causal', type=bool, default=True)
    parser.add_argument('--loss_type', type=str, default='gan')
    parser.add_argument('--negative_rate', type=int, default=0.01)
    parser.add_argument('--drop_rate', type=float, default=0.5)
    parser.add_argument('--is_attention', type=bool, default=True)
    parser.add_argument('--is_store', type=bool, default=True)
    parser.add_argument('--pre_train', type=bool, default=False)
    parser.add_argument('--checkpoint_dir', type=str, default='save/' + filename + '/')
    config = parser.parse_args()
    hyperParams = parameters.getHyperParams()
    if config.pre_train == True:
        hyperParams.step_D = 0
    print(config)
    print(hyperParams)

    # get data
    data = Data(config.data_root, config.dataname, config.L)
    num_users = data.num_users
    num_items = data.num_items
    train = data.train_set
    test = data.test_set
    negdata = data.neg_set
    user_set = data.user_set
    realVector = data.realVector

    batchnum = list(range(num_users))
    # config.negative_num = num_items
    # print(config.negative_num)

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    with tf.Session(config=gpu_config) as sess:
        model = DiCGAN(config, hyperParams, num_items, num_users)
        if config.is_store:
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(config.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Restore model from {} successfully!'.format(ckpt.model_checkpoint_path))
            else:
                print('Restore model from {} failed!'.format(config.checkpoint_dir))
                sess.run(tf.global_variables_initializer())
        else:
            sess.run(tf.global_variables_initializer())

        k_list = [5, 10]
        # k_list = [10]
        best_epoch = np.zeros(len(k_list))
        Best_HR = np.zeros(len(k_list))
        Best_Prec = np.zeros(len(k_list))
        Best_Recall = np.zeros(len(k_list))
        Best_F1 = np.zeros(len(k_list))
        Best_NDCG = np.zeros(len(k_list))
        Best_MAP = np.zeros(len(k_list))
        flag = np.zeros(len(k_list))
        step = 0

        fw = open('log/DiCGAN_log_'+filename+'.txt', 'w')

        for epoch in range(hyperParams.epochs):

            batchNum_D = len(batchnum) // hyperParams.batchSize_D + 1
            for epoch_D in range(hyperParams.step_D):
                t0 = time()
                D_loss = 0
                random.shuffle(batchnum)
                for batchID in tqdm(range(batchNum_D - 1), desc='batch_D'):
                    begin = batchID * hyperParams.batchSize_D
                    end = (batchID + 1) * hyperParams.batchSize_D
                    Index = batchnum[begin: end]
                    x = train[Index]

                    trainMaskVector = []
                    for d in Index:
                        tmp =np.copy(realVector[d])
                        unobserved = list(np.where(tmp == 0)[0])
                        sample = np.random.choice(unobserved, int(len(unobserved) * config.negative_rate), replace=False)
                        tmp[sample] = 1.0
                        trainMaskVector.append(tmp)

                    fetches = [model.trainer_D, model.d_loss]
                    feed_dict = {model.X: train[Index], model.user_seq: user_set[Index], model.mask: trainMaskVector, model.real_data: realVector[Index], model.is_train: 1}
                    _, loss_d = sess.run(fetches, feed_dict)
                    D_loss += loss_d
                print('Epoch:{}, epoch_D:{}, D_loss:{}, D_time:{}'.format(epoch, epoch_D, D_loss, time() - t0))

            batchNum_G = len(batchnum) // hyperParams.batchSize_G + 1
            for epoch_G in range(hyperParams.step_G):
                step += 1
                t1 = time()
                G_loss = 0
                random.shuffle(batchnum)
                for batchID in tqdm(range(batchNum_G - 1), desc='batch_G'):
                    begin = batchID * hyperParams.batchSize_G
                    end = (batchID + 1) * hyperParams.batchSize_G
                    Index = batchnum[begin: end]
                    x = train[Index]

                    trainMaskVector = []
                    for d in Index:
                        tmp = np.copy(realVector[d])
                        unobserved = list(np.where(tmp == 0)[0])
                        sample = np.random.choice(unobserved, int(len(unobserved) * config.negative_rate), replace=False)
                        tmp[sample] = 1.0
                        trainMaskVector.append(tmp)

                    neg_dim = []
                    for d in Index:
                        tmp = np.zeros_like(realVector[d])
                        unobserved = list(np.where(tmp == 0)[0])
                        sample = np.random.choice(unobserved, int(len(unobserved) * config.negative_rate), replace=False)
                        tmp[sample] = 1.0
                        neg_dim.append(tmp)

                    if config.pre_train:
                        fetches = [model.trainer, model.loss]
                    else:
                        fetches = [model.trainer_G, model.g_loss]
                    feed_dict = {model.X: train[Index], model.user_seq: user_set[Index], model.mask: trainMaskVector, model.real_data: realVector[Index], model.is_train: 1, model.Neg_dim: neg_dim}
                    _, loss_g = sess.run(fetches, feed_dict)
                    G_loss += loss_g

                print('Epoch:{}, epoch_G:{}, G_loss:{}, G_time:{}'.format(epoch, epoch_G, G_loss, time() - t1))

            if epoch % 1 == 0:
                t2 = time()
                HR, Prec, Recall, F1, NDCG, MAP = evaluate_ranking(model, sess, train, test, negdata, user_set, k_list=k_list)
                test_log = ''
                for k in range(len(k_list)):
                    test_log = test_log + 'epoch:%d, top%d, precision:%.6f, recall:%.6f, ndcg:%.6f\n' % (step, k_list[k], Prec[k], Recall[k], NDCG[k])
                    fw.write(test_log)

                    if Prec[k] > Best_Prec[k]:
                        Best_HR[k] = HR[k]
                        Best_Prec[k] = Prec[k]
                        Best_Recall[k] = Recall[k]
                        Best_F1[k] = F1[k]
                        Best_NDCG[k] = NDCG[k]
                        best_epoch[k] = epoch + 1

                        flag[k] = 0
                        if config.pre_train == True and config.is_store == False:
                            ckpt_path = config.checkpoint_dir + 'model_' + '.ckpt'
                            model.saver.save(sess, ckpt_path, global_step=epoch)
                            print("model saved to {}".format(ckpt_path))
                    else:
                        flag[k] += 1
                print(test_log)
                print('not increased: ', flag)
                fw.flush()

            # early stop
            if min(flag) >= 50:
                break

        str_log = '-----------------Best Result---------------------\n'
        for k in range(len(k_list)):
            str_log = str_log + 'Top-%d:' \
                                '\tEpoch:%d' \
                                '\tHR:%.6f' \
                                '\tPrec:%.6f' \
                                '\tRecall:%.6f' \
                                '\tF1:%.6f' \
                                '\tNDCG:%.6f\n' % (k_list[k], best_epoch[k], Best_HR[k], Best_Prec[k],
                                                  Best_Recall[k], Best_F1[k], Best_NDCG[k])

        print(str_log)
