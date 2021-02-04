"""
    @Time   : 2020.01.16
    @Author : Zhiqiang Guo
    @Email  : zhiqiangguo@hust.edu.cn
    This is a file about data preprocessing.
"""

from tqdm import tqdm
import json
import random
import numpy as np

def datapre(filepath):
    iidofuser = {}
    with tqdm(open(filepath, 'r'), desc='count the interaction of user and item') as f:
        for line in f:
            if line:
                try:
                    lines = line[:-1].split('\t')
                    user = lines[0]
                    item = lines[1]
                    score = float(lines[2])
                    time = int(lines[3])

                    if user not in iidofuser.keys():
                        iidofuser[user] = []
                        iidofuser[user].append([item, score, time])
                    else:
                        iidofuser[user].append([item, score, time])
                except Exception as e:
                    print(e)
                    pass

    datadic = {}
    newUID = {}
    newIID = {}
    i = 0
    j = 0
    size = 0
    for user, list in iidofuser.items():
        if 10 <= len(list) <= L:
            if user not in newUID.keys():
                newUID[user] = i
                i += 1
            for item, score, time in list:
                size += 1
                if item not in newIID.keys():
                    newIID[item] = j
                    j += 1
                if newUID[user] not in datadic.keys():
                    datadic[newUID[user]] = []
                    datadic[newUID[user]].append([newIID[item], score, time])
                else:
                    datadic[newUID[user]].append([newIID[item], score, time])

    return datadic, newUID, newIID, size

def getTrainTest(num_items, datadic, L, rate):
    '''
        80% train, 20% test
        :param datadic: the dictionary of data
        :param rate: the rate of train set
        :return: train set、test set、
    '''

    train_set = {}
    test_set = {}
    neg_set = {}
    for u, lists in tqdm(datadic.items(), desc='get train set and test set'):
        sort_list = sorted(lists, key=lambda x: x[2])
        items = [i[0] for i in sort_list]
        if len(items) > L:
            items = items[:L]

        trainnum = int(len(items) * rate)
        neg_num = (len(items) - trainnum) * 9
        if neg_num < 100:
            neg_num = 100
        if neg_num > num_items:
            neg_num = num_items - len(items)

        train = items[:trainnum]
        train_add = np.ones(int(L * rate) - len(train)) * num_items
        # while len(train) < L * rate:
        #     # train.append(random.choice(items[:trainnum]))
        #     train.append(num_items)
        train_set[u] = list(map(int, list(train_add))) + train
        test_set[u] = items[trainnum:]

        neg_chooses = set(range(num_items)) - set(items)
        neg_chooses = list(neg_chooses)
        neg_list = random.sample(neg_chooses, neg_num)
        neg_set[u] = neg_list

    return train_set, test_set, neg_set


if __name__ == "__main__":
    L = 100
    filename = 'Ciao'
    path = 'datasets/'+ filename + '/'
    filepath = path + 'ratings_' + filename +'.txt'

    datadic, uiddic, iiddic, size = datapre(filepath)

    num_users = len(uiddic.keys())
    num_items = len(iiddic.keys())

    print("Loading Success!\n"
          "Data Info:\n"
          "\tUser Num: {}\n"
          "\tItem Num: {}\n"
          "\tData Size: {}\n"
          "\tSparsity: {}\n".format(num_users, num_items, size,
                                    1 - (size / (num_users * num_items))))

    train_set, test_set, neg_set = getTrainTest(num_items, datadic, L, 0.8)

    file_uid = open(path + filename + '_uid.txt', 'w')
    file_uid.write(json.dumps(uiddic))

    file_iid = open(path + filename + '_iid.txt', 'w')
    file_iid.write(json.dumps(iiddic))

    file_train = open(path + filename + '_train.txt', 'w')
    file_train.write(json.dumps(train_set))

    file_test = open(path + filename + '_test.txt', 'w')
    file_test.write(json.dumps(test_set))

    # file_neg = open(path + filename + '_neg.txt', 'w')
    # file_neg.write(json.dumps(neg_set))