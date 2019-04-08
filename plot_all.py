'''
Script to plot all runs in a folder.
Say you have

data-trial/ppo/Pendulum-v0/190312_140449
data-trial/ppo/Pendulum-v0/190412_141241
...

This script will plot one line (for the desired data) for each run.
Each line will have a legend entry with the filename.

You can then press R to refresh the plot (e.g., if some trials are still running)
or ESC to close plot and end the program. 
'''

import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib
plt.ion()
matplotlib.use('TKagg')
import seaborn as sns
import itertools
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=int, help='index of the column to plot', required=True)
    parser.add_argument('--f', type=str, help='folder where .dat files are stored', required=True)

    args = parser.parse_args()
    folder = args.f
    col = args.c

    def update():
        plt.cla()
        palette = itertools.cycle(sns.color_palette())
        l = []
        for f in os.listdir(folder):
            if f.endswith(".dat"):
                data_mat = np.loadtxt(os.path.join(folder, f))
                if data_mat.shape[0] > 0:
                    l.append(f)
                    color = next(palette)
                    data = data_mat[:,col]
                    data = data[np.logical_and(~np.isnan(data), ~np.isinf(data))]
                    plt.plot(data, color=color)
        leg = plt.legend(handles=plt.gca().lines, labels=l, loc='lower left')
        frame = leg.get_frame()
        frame.set_facecolor('white')
        plt.draw()
        print('refreshed')

    def handle(event):
        if event.key == 'r':
            update()
        if event.key == 'escape':
            sys.exit(0)

    sns.set_context("paper")
    sns.set_style('darkgrid', {'legend.frameon':True})

    fig = plt.figure()
    plt.axes()
    picsize = fig.get_size_inches() / 1.3
    fig.set_size_inches(picsize)
    fig.canvas.mpl_connect('key_press_event', handle)
    update()

    input('')
