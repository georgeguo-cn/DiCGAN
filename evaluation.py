"""
    @Time   : 2020.01.16
    @Author : Zhiqiang Guo
    @Email  : zhiqiangguo@hust.edu.cn
    This is a file about evaluation.
"""

import math
import numpy as np
from tqdm import tqdm
from time import time

def _compute_ndcg_map(targets, prediction, k):

    if len(prediction) >= k:
        pred = prediction[:k]

    dcg = 0.0
    idcg = 0.0
    ndcg = 0.0
    ap = 0.0
    map = 0.0
    j = 0

    for i, p in enumerate(pred):
        if p in targets:
            dcg += 1.0/math.log2(i + 2)
            idcg += 1.0/math.log2(j + 2)
            j += 1
            ap += (j + 1) / (i + 1)

    if not list(targets):
        ndcg = 0.0
        map = 0.0

    if idcg == 0.0:
        ndcg = 0.0

    if ap > 0.0:
        map = ap / len(targets)

    if idcg > 0.0:
        ndcg = dcg / idcg

    return ndcg, map


def _compute_hr_precision_recall(targets, prediction, k):

    if len(prediction) >= k:
        pred = prediction[:k]
    num_hit = len(set(pred).intersection(set(targets)))

    hr, prec, recall, f1 = 0.0, 0.0, 0.0, 0.0
    if num_hit != 0:
        hr = 1.0
        prec = float(num_hit) / k
        recall = float(num_hit) / len(targets)
        f1 = (2 * prec * recall) / (prec + recall + 1e-20)
    return hr, prec, recall, f1

def evaluate_ranking(model, sess, train, test, neg_data, user_set, k_list):

    HR_list = np.zeros(len(k_list))
    Prec_list = np.zeros(len(k_list))
    Recall_list = np.zeros(len(k_list))
    F1_list = np.zeros(len(k_list))
    NDCG_list = np.zeros(len(k_list))
    MAP_list = np.zeros(len(k_list))

    predictions = sess.run([model.G_output], {model.X: train, model.user_seq: user_set, model.is_train: 0})[0]
    for j in tqdm(range(len(train)), desc='Test：'):

        result = predictions[j]
        mask = np.zeros(len(result))
        mask[test[j]] = 1
        mask[neg_data[j]] = 1
        result = result * mask
        sort_index = (-result).argsort()
        prediction = list(sort_index[:k_list[-1]])
        for k in range(len(k_list)):
            hr, prec, recall, f1 = _compute_hr_precision_recall(test[j], prediction, k_list[k])
            ndcg, map = _compute_ndcg_map(test[j], prediction, k_list[k])
            HR_list[k] += hr
            Prec_list[k] += prec
            Recall_list[k] += recall
            F1_list[k] += f1
            NDCG_list[k] += ndcg
            MAP_list[k] += map

    HR_list = HR_list / len(test)
    Prec_list = Prec_list / len(test)
    Recall_list = Recall_list / len(test)
    F1_list = F1_list / len(test)
    NDCG_list = NDCG_list / len(test)
    MAP_list = MAP_list / len(test)

    return HR_list, Prec_list, Recall_list, F1_list, NDCG_list, MAP_list