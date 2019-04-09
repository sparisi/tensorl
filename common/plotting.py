import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
plt.ion()
matplotlib.use('TKagg')

import numpy as np

class My3DPlot:
    '''
    It creates a new figure with two subfigures (surf and contourf + colorbar)
    and allows to update them with new z-values without creating a new plot.
    '''

    def __init__(self, xmin, xmax, ymin, ymax, title, n=10):
        self.name = 'my3dplot'
        self.fig = plt.figure()
        self.fig.suptitle(title)
        self.n = n
        self.ax_surf = self.fig.add_subplot(121, projection='3d')
        self.ax_contour = self.fig.add_subplot(122)
        x = np.linspace(xmin, xmax, n)
        y = np.linspace(ymin, ymax, n)
        self.xx, self.yy = np.meshgrid(x, y)
        self.XY = np.vstack((self.xx.flatten(),self.yy.flatten())).T # shape is (n^2,2)
        self.surf = self.ax_surf.plot_surface(self.xx, self.yy, np.zeros((n,n)), cmap=cm.coolwarm)
        self.contour = self.ax_contour.contourf(self.xx, self.yy, np.zeros((n,n)), cmap=cm.coolwarm)
        self.ax_cbar = plt.colorbar(self.contour).ax

    def update(self, z):
        self.fig.canvas.flush_events()
        self.ax_surf.cla()
        self.ax_contour.cla()
        self.ax_cbar.cla()
        self.surf = self.ax_surf.plot_surface(self.xx, self.yy, z.reshape((self.n,self.n)), cmap=cm.coolwarm)
        self.contour = self.ax_contour.contourf(self.xx, self.yy, z.reshape((self.n,self.n)), cmap=cm.coolwarm)
        plt.colorbar(self.contour, cax=self.ax_cbar)
        plt.draw()



class My2DScatter:
    '''
    It creates a new figure with a scatter plot and allows to update it with new
    values without creating a new plot.
    '''

    def __init__(self, title):
        self.name = 'my2dscatter'
        self.fig, self.ax = plt.subplots()
        self.fig.suptitle(title)
        self.scatter = self.ax.scatter([],[])

    def update(self, x, y, value):
        self.fig.canvas.flush_events()
        self.ax.cla()
        self.scatter = self.ax.scatter(x, y, c=value)
        plt.draw()
