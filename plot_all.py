'''
Script to plot all runs in a folder.
Say you have

data-trial/ppo/Pendulum-v0/190312_140449
data-trial/ppo/Pendulum-v0/190412_141241
...

This script will plot one line (for the desired data) for each run.
Each line will have a legend entry with the filename.

You can then press R to refresh the plot (e.g., if some trials are still running)
or P to save the plot as pdf,
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
    parser.add_argument('--col', type=int, help='index of the column to plot', required=True)
    parser.add_argument('--src', type=str, help='folder where .dat files are stored', required=True)
    parser.add_argument('--pdf', type=str, help='(optional) filename to save as pdf', required=False)

    args = parser.parse_args()
    folder = args.src
    col = args.col

    def savepdf():
        if args.pdf is not None:
            plt.savefig(args.pdf+".pdf", bbox_inches='tight', pad_inches=0)

    def update():
        plt.cla()
        palette = itertools.cycle(sns.color_palette())
        lines = itertools.cycle(["-","--","-.",":"])
        l = []
        for f in sorted(os.listdir(folder)):
            if f.endswith(".dat"):
                data_mat = np.loadtxt(os.path.join(folder, f))
                if data_mat.shape[0] > 0:
                    try:
                        data = data_mat[:,col]
                        l.append(f)
                    except:
                        print('Cannot read', f)
                        continue
                    data = data[np.logical_and(~np.isnan(data), ~np.isinf(data))]
                    plt.plot(data, color=next(palette), linestyle=next(lines))
        leg = plt.legend(handles=plt.gca().lines, labels=l, loc='best')
        frame = leg.get_frame()
        frame.set_facecolor('white')
        plt.draw()

    def handle(event):
        if event.key == 'r' or event.key == 'R':
            update()
            print('refreshed')
        if event.key == 'p' or event.key == 'P':
            savepdf()
            print('saved as pdf')
        if event.key == 'escape':
            print('quit')
            sys.exit(0)

    def handle_close(event):
        sys.exit(0)

    sns.set_context("paper")
    sns.set_style('darkgrid', {'legend.frameon':True})

    fig = plt.figure()
    plt.axes()
    picsize = fig.get_size_inches() / 1.3
    fig.set_size_inches(picsize)
    fig.canvas.mpl_connect('key_press_event', handle)
    fig.canvas.mpl_connect('close_event', handle_close)
    update()
    savepdf()

    input('')
