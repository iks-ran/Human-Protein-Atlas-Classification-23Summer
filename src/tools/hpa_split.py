import pandas as pd
import os
import numpy as np
from skmultilearn.model_selection import IterativeStratification

data_path = 'data/HPA'
val_split = 0.3
num_labels = 28

def iterative_train_test_split(X, y, test_size):

    stratifier = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[test_size, 1.0-test_size])
    train_indexes, test_indexes = next(stratifier.split(X, y))

    return train_indexes, test_indexes


def split(path, val_split=0.3):
    samples = pd.read_csv(f'{data_path}/train.csv')
    sample_ids = samples.iloc[:, 0].values
    sample_labels = samples.iloc[:, 1].values

    sample_vectors = np.zeros(shape=(len(sample_ids), num_labels), dtype=np.int32)
    for i in range(len(sample_ids)):
        sample_vectors[i][[int(l) for l in sample_labels[i].split(' ')]] = 1

    train_ids, val_ids = iterative_train_test_split(sample_ids, sample_vectors, test_size=val_split)

    return sample_ids[train_ids], sample_labels[train_ids], sample_ids[val_ids], sample_labels[val_ids]


if __name__ == '__main__': 
    id_train, label_train, id_val, label_val = split(data_path, val_split)
    train = pd.DataFrame({'Id': id_train, 
                          'Target': label_train})
    val = pd.DataFrame({'Id': id_val, 
                        'Target': label_val})
    train.to_csv(f'{data_path}/train_split.csv', index=False)
    val.to_csv(f'{data_path}/val_split.csv', index=False)
