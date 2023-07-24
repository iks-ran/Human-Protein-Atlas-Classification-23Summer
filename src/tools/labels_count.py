import json
import argparse
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default='./data/HPA', 
                        help='Path of data')
    parser.add_argument('-s', '--split', type=str, default='train')
    parser.add_argument('-j', '--json', type=str, default='./data/HPA/label2cat.json')
    args = parser.parse_args()
    
    samples = pd.read_csv(f"{args.path}/{args.split}.csv")
    with open(args.json, 'r') as f:
        data = f.read()
    label2cat = json.loads(data)
    num_labels = len(label2cat)
    sample_labels = samples.iloc[:, 1].values
    sample_ids = samples.iloc[:, 0].values

    sample_vectors = np.zeros(shape=(len(sample_ids), num_labels), dtype=np.int32)
    for i in range(len(sample_ids)):
        sample_vectors[i][[int(l) for l in sample_labels[i].split(' ')]] = 1
    
    labels_count = []
    categories = []
    for label, cat in label2cat.items():
        cat_count = sample_vectors[:, int(label)].sum()
        labels_count.append(cat_count)
        categories.append(cat)

    sorted_data = sorted(enumerate(labels_count), key=lambda x: x[1], reverse=True)
    sorted_labels_count = [x[1] for x in sorted_data]
    sorted_indices = [x[0] for x in sorted_data]
    sorted_cats = [categories[i] for i in sorted_indices]

    # fontdict = {'family': 'serif', 'color': 'darkblue', 'weight': 'bold', 'size': 12}
    fontdict=dict(fontsize=16,
              color='g',
              family='serif',
              weight='light',
              style='italic',
              )
    plt.figure(figsize=(12, 7))
    plt.style.use('seaborn-darkgrid')
    plt.bar(sorted_cats, sorted_labels_count, color='skyblue', alpha=0.8)
    plt.xlabel('Category', fontdict=fontdict)
    plt.ylabel('Samples Count', fontdict=fontdict)
    plt.title(f'Count Histogram of Each Label - {args.split}', fontdict=fontdict)
    plt.ylim(0, max(sorted_labels_count) * 1.1)
    for i, v in enumerate(sorted_labels_count):
        plt.text(i, v, f"{v}", ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(f'{args.path}/{args.split}_count.png')

if __name__ == "__main__":
    main()