import re
import argparse
import os
import matplotlib.pylab as plt
import numpy as np

pattern_train = r"Epoch: \[ (\d+) / (?:\d+) \] Loss: (\d+\.\d+) Best Validation Loss "
pattern_val = r"Epoch \[ (\d+) / (?:\d+) \] Validation Loss: (\d+\.\d+) Best Validation Loss: (?:\d+\.\d+) F1: (\d+\.\d+) Best F1:"
model_pattern  =r"exp/(.+)/(?:\d+)-"

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log', type=str, help='Path of log')

    args = parser.parse_args()
    with open(args.log, 'r') as f:
        log = f.read()
    train_log = re.findall(pattern_train, log)
    val_log = re.findall(pattern_val, log)
    model = re.findall(model_pattern, args.log)[0]
    # print(val_log)

    val_epochs = np.asarray([int(i[0]) for i in val_log])
    val_losses = np.asarray([float(i[1]) for i in val_log])
    val_f1 = np.asarray([float(i[2]) for i in val_log])
    train_epochs = np.asarray([int(i[0]) for i in train_log])
    train_losses = np.asarray([float(i[1]) for i in train_log])

    fontdict=dict(fontsize=16,
              # color='g',
              family='serif',
              weight='light',
              style='italic',
              )

    fig, ax1 = plt.subplots(figsize=(12, 7))
    plt.style.use('seaborn-paper')
    ax2 = ax1.twinx()
    ax1.plot(train_epochs, train_losses, label="train_loss", linewidth=1, color='tab:blue')
    ax1.scatter([val_epochs], train_losses[val_epochs - 1], marker='*', s=40, color='orange')
    ax1.scatter([1], train_losses[0], marker='*', s=40, color='orange')
    ax1.plot(val_epochs, val_losses, label="val_loss", marker='*', markersize=8, markerfacecolor='green', linewidth=1, color='tab:orange')
    ax2.plot(val_epochs, val_f1, label="val_f1", marker='^', markersize=8, markerfacecolor='green', linewidth=1, color='tab:orange')
    ax1.set_xlabel('Epochs', fontdict=fontdict)
    ax1.set_ylabel('Loss', fontdict=fontdict)
    ax2.set_ylabel('F1', fontdict=fontdict)
    ax2.set_ylim([0, 1])
    ax1.set_title(f'{model}  Loss&F1 - Epoch Curve', fontdict=fontdict)
    ax1.legend(loc='lower left')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f'{os.path.dirname(args.log)}/loss.png')

if __name__ == '__main__':
    main()
