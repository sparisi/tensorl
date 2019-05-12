'''
Script to plot data generated with one of the run scripts (run_slurm, run_multiproc, run_joblib).
It first loads .dat files and read data according to the specified column index.
Then it plots the (moving) average with 95% confidence interval.

Change style, legend position, x ticks, ... according to your needs.
'''

import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import itertools


def moving_average(data, window=4):
    return np.convolve(data, np.ones(int(window)) / float(window), 'same')

def shaded_plot(ax, data, **kw):
    x = np.arange(data.shape[1])
    mu = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    ci = 1.96*std/np.sqrt(data.shape[0])
    ax.fill_between(x, mu-ci, mu+ci, alpha=0.2, edgecolor="none", linewidth=0, **kw)
    ax.plot(x, mu, **kw)
    ax.margins(x=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=int, help='index of the column to plot', required=True)
    parser.add_argument('--e', type=str, help='name of the environment', required=True)
    parser.add_argument('--a', nargs='+', type=str, help='list of the algorithms', required=True)
    parser.add_argument('--f', type=str, help='data folder', default='data-trial/')
    parser.add_argument('--t', type=str, help='title', default='')
    parser.add_argument('--x', type=str, help='x-axis label', default='')
    parser.add_argument('--y', type=str, help='y-axis label', default='')
    parser.add_argument('--l', type=str, help='legend names (if empty, the algorithms names will be used)', default='')
    parser.add_argument('--m', type=int, help='moving average window (default 1, not used)', default='1')


    args = parser.parse_args()
    if not args.l:
        args.l = args.a

    sns.set_context("paper")
    sns.set_style('darkgrid', {'legend.frameon':True})

    fig = plt.figure()
    plt.axes()
    ax = fig.add_subplot(111)
    plt.xlabel(args.x)
    plt.ylabel(args.y)
    plt.title(args.t)

    palette = itertools.cycle(sns.color_palette())

    for alg in args.a:
        color = next(palette)
        path = os.path.join(args.f, alg, args.e)
        data = []
        for i in range(5):
            data_file = os.path.join(path, str(i) + '.dat')
            if not os.path.exists(data_file):
                print('Missing file! ' + data_file)
                continue
            data_mat = np.loadtxt(data_file)
            data.append(data_mat[:,args.c])
        data = np.array(data)

        if data.shape[0] == 0:
            continue

        if args.m > 1:
            for i in range(data.shape[0]):
                data[i,:] = moving_average(data[i,:], args.m)

        shaded_plot(ax, data, color=color)

    # CHANGE IT!
    # plt.xticks([0, 2000, 4000, 6000, 8000, 10000], ['0', '2', '4', '6', '8', '10'])
    leg = plt.legend(handles=ax.lines, labels=args.l, loc='best')

    frame = leg.get_frame()
    frame.set_facecolor('white')
    picsize = fig.get_size_inches() / 1.3
    fig.set_size_inches(picsize)
    plt.savefig(args.e+args.c+".pdf", bbox_inches='tight', pad_inches=0)
