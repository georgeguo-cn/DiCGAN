"""
    @Time   : 2020.01.16
    @Author : Zhiqiang Guo
    @Email  : zhiqiangguo@hust.edu.cn
    This file is used to load datasets.
"""

from tqdm import tqdm
import numpy as np
import random
import json

class Data():

    def __init__(self, path, filename, L):

        self.L = L
        self.path = path
        self.filename = filename
        with open(path + filename + '_uid.txt', 'r') as f:
            uiddic = json.loads(f.read())
        with open(path + filename + '_iid.txt', 'r') as f:
            iiddic = json.loads(f.read())

        self.num_users = len(uiddic.keys())
        self.num_items = len(iiddic.keys())

        self.train_set, self.test_set, self.neg_set, self.realVector, self.user_set = self.getTrainTest()

    def getTrainTest(self):

        '''
            80% train, 20% test
            :param datadic: the dictionary of data
            :param rate: the rate of train set
            :return: train set、test set、
        '''
        with open(self.path + self.filename + '_train.txt', 'r') as f:
            train_dic = json.loads(f.read())
        with open(self.path + self.filename + '_test.txt', 'r') as f:
            test_dic = json.loads(f.read())
        with open(self.path + self.filename + '_neg.txt', 'r') as f:
            neg_dic = json.loads(f.read())
        train_set = []
        test_set = []
        user_set = []
        realVector = []
        neg_set = []
        for u, list in train_dic.items():
            user_set.append([u])
            train_set.append(list)
            test_set.append(test_dic[u])
            neg_set.append(neg_dic[u])

            vector = [0] * self.num_items
            for i in list:
                if i < self.num_items:
                    vector[i] = 1.0
            realVector.append(vector)

        return np.array(train_set), np.array(test_set), np.array(neg_set), np.array(realVector), np.array(user_set)