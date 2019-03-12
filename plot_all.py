'''
Script to plot all runs in a folder.
Say you have

data-trial/ppo/Pendulum-v0/190312_140449
data-trial/ppo/Pendulum-v0/190412_141241
...

This script will plot one line (for the desired data) for each run.
Each line will have a legend entry with the filename.
'''

import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=int, help='index of the column to plot')
    parser.add_argument('--f', type=str, help='folder where .dat files are stored')

    args = parser.parse_args()

    sns.set_context("paper")
    sns.set_style('darkgrid', {'legend.frameon':True})

    fig = plt.figure()
    plt.axes()
    ax = fig.add_subplot(111)

    palette = itertools.cycle(sns.color_palette())

    l = []
    for f in os.listdir(args.f):
        if f.endswith(".dat"):
            data_mat = np.loadtxt(os.path.join(args.f, f))
            if data_mat.shape[0] > 0:
                l.append(f)
                color = next(palette)
                data = data_mat[:,args.c]
                data = data[np.logical_and(~np.isnan(data), ~np.isinf(data))]
                ax.plot(data, color=color)

    leg = plt.legend(handles=ax.lines, labels=l, loc='lower left')
    frame = leg.get_frame()
    frame.set_facecolor('white')
    picsize = fig.get_size_inches() / 1.3
    fig.set_size_inches(picsize)

    plt.show()
