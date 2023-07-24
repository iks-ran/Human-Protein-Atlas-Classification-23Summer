import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score

def logits2label(logits, thresh=0.5):
    return 1 * (logits > thresh)

def F1(labels, pred, average='weighted'):
    return f1_score(labels, pred, average=average)

def Recall(labels, pred, average='weighted'):
    return recall_score(labels, pred, average=average)

def Precision(labels, pred, average='weighted'):
    return precision_score(labels, pred, average=average)