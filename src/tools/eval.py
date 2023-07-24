import json
import argparse
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append("/remote-home/zhoutao/haizhou/src/hpa/src")
from libs.metrics import logits2label, Recall, Precision, F1


def eval(args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--predict', type=str, help='Path of Predict File')
    parser.add_argument('-t', '--truth', type=str, default='./data/HPA/val_split.csv', help='Path of Truth File')
    parser.add_argument('-j', '--json', type=str, default='./data/HPA/label2cat.json')
    if args:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()
    
    PREDICT_LOGITS = pd.read_csv(args.predict)
    TRUTH = pd.read_csv(args.truth)
    with open(args.json, 'r') as f:
        data = f.read()
    label2cat = json.loads(data)
    num_labels = len(label2cat)
    truth_labels = TRUTH.iloc[:, 1].values
    predict_logits = PREDICT_LOGITS.to_numpy()
    pred = logits2label(predict_logits)

    truth = np.zeros(shape=(len(truth_labels), num_labels), dtype=np.int32)
    for i in range(len(truth_labels)):
        truth[i][[int(l) for l in truth_labels[i].split(' ')]] = 1

    cats = [*list(label2cat.values()), 'Average']
    metrics = ["F1", "Recall", "Precision"]
    f1 = F1(truth, pred, average=None)
    recall = Recall(truth, pred, average=None)
    precision = Precision(truth, pred, average=None)
    scores = np.stack([f1, recall, precision], axis=1)
    print(scores.shape)
    av_f1, av_rc, av_pc = F1(truth, pred), Recall(truth, pred), Precision(truth, pred)
    av_scores = np.asarray([av_f1, av_rc, av_pc]).reshape(-1, 3)
    scores = np.concatenate([scores, av_scores], axis=0)
    print(scores.shape)
    dp = pd.DataFrame(scores, index=cats, columns=metrics).round(4)
    print(dp.to_string())
    with open(f"{os.path.dirname(args.predict)}/eval.txt", 'w') as f:
        f.write(dp.to_string()+"\n\n")
    print(f"Save result to {os.path.dirname(args.predict)}/eval.txt")
    return av_f1
    

if __name__ == "__main__":
    eval()