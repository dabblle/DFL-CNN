import argparse
import glob
import math
import os
import random
import shutil
from pathlib import Path
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
plt.switch_backend('agg')



def plot_results(start=0, stop=0, bucket='', id=(), dir_path='./'):  # from utils.utils import *; plot_results()
    # Plot training results files 'results*.txt'
    fig, ax = plt.subplots(2, 5, figsize=(14, 7))
    ax = ax.ravel()
    s = ['GIoU', 'Objectness', 'Classification', 'Precision', 'Recall',
         'val GIoU', 'val Objectness', 'val Classification', 'mAP@0.5', 'F1']
    if bucket:
        os.system('rm -rf storage.googleapis.com')
        files = ['https://storage.googleapis.com/%s/results%g.txt' % (bucket, x) for x in id]
    else:
        files = glob.glob(os.path.join(dir_path, 'results*.txt')) + glob.glob('../../Downloads/results*.txt')
    for f in sorted(files):
        results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
        n = results.shape[1]  # number of rows
        x = range(start, min(stop, n) if stop else n)
        for i in range(10):
            y = results[i, x]
            if i in [0, 1, 2, 5, 6, 7]:
                y[y == 0] = np.nan  # dont show zero loss values
                # y /= y[0]  # normalize
            ax[i].plot(x, y, marker='.', label=Path(f).stem)
            ax[i].set_title(s[i])
            if i in [5, 6, 7]:  # share train and val loss y axes
                ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])

    fig.tight_layout()
    ax[1].legend()
    fig.savefig(os.path.join(dir_path, 'results.png'), dpi=200)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='./')
    opt = parser.parse_args()

    plot_results(dir_path=opt.dir)
