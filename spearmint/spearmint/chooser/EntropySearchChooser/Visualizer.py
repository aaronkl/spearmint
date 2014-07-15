'''
Created on 10.11.2013

@author: Aaron Klein
'''
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from .entropy import Entropy
from .entropy_with_costs import EntropyWithCosts
from .support import compute_expected_improvement, compute_pmin_bins


class Visualizer():

    def __init__(self, index, path='.'):
        '''
        Default constructor.
        Args:
            index: number of the image file
            path: where to store the images
        '''
        self._index = index
        self._costs = index
        self._path = path

    def plot(self, X, y, model, cost_model, cands):

        #fig = plt.figure(figsize=(12, 8))
        #ax = plt.subplot(111)

        num_plots_x = 2
        num_plots_y = 1

        fig, (self._ax1, self._ax2) = plt.subplots(num_plots_x, num_plots_y)

        plt.hold(True)

        self.plot_gp(X, y, model)
        num_of_cands = 101
        entropy_estimator = Entropy(model)
        entropy_plot = np.zeros([num_of_cands])
        points_plot = np.linspace(0, 1, num_of_cands)

        for i in xrange(0, num_of_cands):
            cand = np.array([points_plot[i]])
            entropy_plot[i] = entropy_estimator.compute(cand)

        self.plot_entropy_one_dim(points_plot, entropy_plot)

        ei_values = np.zeros([num_of_cands])
        ei_points = np.linspace(0, 1, num_of_cands)

        for i in xrange(0, num_of_cands):
            ei_values[i] = compute_expected_improvement(np.array([ei_points[i]]), model)

        self.plot_expected_improvement(ei_points, ei_values)

        self.plot_points(entropy_estimator._representer_points)
        #self.plot_points(cands)

        sample_locs = entropy_estimator._representer_points

        #pmin = entropy_estimator._
        grid = np.linspace(0, 1, 100)[:, np.newaxis]
        grid = sample_locs
        pmin = np.zeros(grid.shape[0])
        mean, L = model.getCholeskyForJointSample(grid)

        Omega = np.asfortranarray(np.random.normal(0, 1, (500,
                                                 grid.shape[0])))

        for omega in Omega:
            vals = model.drawJointSample(mean, L, omega)
            mins = np.where(vals == vals.min())[0]
            number_of_mins = len(mins)
            for m in mins:
                pmin[m] += 1. / (number_of_mins)

        pmin = pmin / 500

        self.plot_pmin(pmin, grid)

        self.plot_ground_truth()
        self._ax1.axis([0, 1, -0.05, 0.3])
        self._ax2.axis([0, 1, -2.0, 2.0])
        self._ax2.set_yticks([])
        self._ax1.legend((self._p_gp[0], self._p_comp[0], self._p_ground_truth[0]),
                    ('Performance', 'Observation', "Ground Truth"), loc=1)
        self._ax2.legend((self._p_ei[0], self._p_ent[0],
                    self._p_pmin[0], self._p_rp[0]),
                    ('Expected Improvement', 'Entropy',
                        'Pmin', 'Representer Points'), loc=1)

        filename = self._path + "/plot_" + str(self._index) + ".png"
        self._index += 1

#         box = ax.get_position()
#         ax.set_position([box.x0, box.y0 + box.height * 0.1,
#                      box.width, box.height * 0.9])
#
#         ax.legend((self._p_ei[0], self._p_ent[0], self._p_gp[0],
#                   self._p_pmin[0], self._p_rp[0], self._p_comp[0]),
#                     ('Expected Improvement', 'Entropy', 'Performance',
#                         'Pmin', 'Selected Candidates', 'Comp'),
#                         loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

        plt.savefig(filename)

    def plot_ground_truth(self):
        x = np.linspace(0, 1, 100)[:, np.newaxis]
        y = ((x - 0.5) ** 2)
        self._p_ground_truth = self._ax1.plot(x, y, 'b--')

    def plot_expected_improvement(self, cands, ei_values):
        ei_values = (ei_values - np.mean(ei_values)) / np.sqrt(np.var(ei_values))
        self._p_ei = self._ax2.plot(cands, ei_values, 'g')

    def plot_points(self, points):

        self._p_rp = self._ax2.plot(points, np.ones(points.shape[0]) * (-2), 'ro')

    def plot_entropy_one_dim(self, cands, entropy_values):
        entropy_values = (entropy_values - np.mean(entropy_values)) / np.sqrt(np.var(entropy_values))

        self._p_ent = self._ax2.plot(cands, entropy_values, 'b')

    def plot_pmin(self, pmin, sample_locs):

        self._p_pmin = self._ax2.bar(sample_locs, pmin * 10, bottom=-2,
                               width=0.01, color='k')

    def plot_gp(self, X, y, model):

        plt.hold(True)
        self._p_comp = self._ax1.plot(X, y, 'k+')
        x = np.linspace(0, 1, 100)[:, np.newaxis]

        test_inputs = np.ones((100, 1))
        for i in range(0, 100):
            test_inputs[i] = x[i]

        (mean, variance) = model.predict(test_inputs, True)
        lower_bound = np.zeros((100, 1))
        for i in range(0, mean.shape[0]):
            lower_bound[i] = mean[i] - math.sqrt(variance[i])

        upper_bound = np.zeros((100, 1))
        for i in range(0, mean.shape[0]):
            upper_bound[i] = mean[i] + math.sqrt(variance[i])

        self._p_gp = self._ax1.plot(x, mean, 'r')
        self._ax1.fill_between(test_inputs[:, 0], upper_bound[:, 0],
                         lower_bound[:, 0], facecolor='red', alpha=0.6)

    def plot3D(self, X, y, model, cost_model, next_cand, cand, points_per_axis=50,
               entropy_values=None, incumbent=None, rep=None):

        self._points_per_axis = points_per_axis
        self._fig = plt.figure()
        plt.hold(True)
        fig = plt.figure(figsize=plt.figaspect(0.5))

        #the number of plots in x and y direction
        num_plots_x = 2
        num_plots_y = 2
        axis_index = 1

        self._ax = fig.add_subplot(num_plots_x, num_plots_y, axis_index, projection='3d')
        self._ax.text2D(0.0, 0.1, "Model")
        axis_index += 1

        self._ax_2 = fig.add_subplot(num_plots_x, num_plots_y, axis_index, projection='3d')
        self._ax_2.text2D(0.0, 0.1, "Entropy")
        axis_index += 1

        self._ax_3 = fig.add_subplot(num_plots_x, num_plots_y, axis_index)
        self._ax_3.text(0.25, 1.05, "Expected Improvement")
        axis_index += 1

        self._ax_4 = fig.add_subplot(num_plots_x, num_plots_y, axis_index)
        self._ax_4.text(0.45, 1.05, "Pmin")
        axis_index += 1

        self._ax_4.plot(rep[:, 0], rep[:, 1], 'ko', label="Representers")
        self._ax_4.plot(rep[-1, 0], rep[-1, 1], 'ms', label="Incumbent")

        self._ax_3.axis([0, 1, 0, 1])
        self._ax_4.axis([0, 1, 0, 1])

        self._ax_3.set_xlabel('S')
        self._ax_3.set_ylabel('X')
        self._ax_4.set_xlabel('S')
        self._ax_4.set_ylabel('X')

        self.plot3D_gp(model)

        x = np.linspace(0, 1, points_per_axis)[:, np.newaxis]
        y = np.linspace(0, 1, points_per_axis)[:, np.newaxis]

        x, y = np.meshgrid(x, y)

        test_inputs = np.zeros((points_per_axis * points_per_axis, 2))

        ei_values = np.zeros([points_per_axis, points_per_axis])

        for i in xrange(0, points_per_axis):
            for j in xrange(0, points_per_axis):
                test_inputs[i * points_per_axis + j, 0] = x[j][i]
                test_inputs[i * points_per_axis + j, 1] = y[j][i]
                ei_values[j][i] = compute_expected_improvement(np.array([x[i][j], y[i][j]]), model)

        self._ax_3.hold(True)
        self._ax_3.plot(X[:, 0], X[:, 1], 'ro', label="Points")
        self._ax_3.plot(cand[:, 0], cand[:, 1], 'b+', label="Candidates")

        self.plot3D_entropy(cand[:, 0], cand[:, 1], entropy_values)

        self._ax_3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

        #self._ax_3.contour(x, y, ei_values, colors=('yellow'))
        self._ax_3.pcolor(x, y, ei_values, cmap='Greens')

        self._ax_4.hold(True)

        x_idx = np.linspace(0, 1, points_per_axis)[:, np.newaxis]
        y_idx = np.linspace(0, 1, points_per_axis)[:, np.newaxis]

        x, y = np.meshgrid(x_idx, y_idx)
        grid = np.zeros((points_per_axis * points_per_axis, 2))
        for i in xrange(0, points_per_axis):
            for j in xrange(0, points_per_axis):
                grid[i * points_per_axis + j, 0] = x[j][i]
                grid[i * points_per_axis + j, 1] = y[j][i]

        pmin = np.zeros(grid.shape[0])
        mean, L = model.getCholeskyForJointSample(grid)

        Omega = np.asfortranarray(np.random.normal(0, 1, (500,
                                                 grid.shape[0])))

        for omega in Omega:
            vals = model.drawJointSample(mean, L, omega)
            mins = np.where(vals == vals.min())[0]
            number_of_mins = len(mins)
            for m in mins:
                pmin[m] += 1. / (number_of_mins)
            pmin += pmin / 500

        hist = np.zeros([points_per_axis, points_per_axis])

        for i in xrange(0, points_per_axis):
            for j in xrange(0, points_per_axis):
                hist[j][i] = pmin[i * points_per_axis + j]

        self._ax_4.axis([0, 1, 0, 1])

        self._ax_4.contour(x, y, hist, colors=('blue'), label="Pmin")

        self._ax_4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
        filename = self._path + "/plot3D_" + str(self._index) + ".png"
        self._index += 1

        plt.savefig(filename)

        #compute pmin
#         representers = np.ones([points_per_axis, 2])
#         representers[:, 1] = np.linspace(0, 1, points_per_axis)[:]
#         Omega = np.asfortranarray(np.random.normal(0, 1, (100,
#                                          points_per_axis)))
#         mean, L = model.getCholeskyForJointSample(representers)
#         pmin = compute_pmin_bins(Omega, mean, L)
# 
#         self._points_per_axis = points_per_axis
#         self._fig = plt.figure()
#         fig = plt.figure(figsize=plt.figaspect(0.5))
# 
#         #the number of plots in x and y direction
#         num_plots_x = 2
#         num_plots_y = 4
#         axis_index = 1
# 
#         self._ax = fig.add_subplot(num_plots_x, num_plots_y, axis_index, projection='3d')
#         self._ax.text2D(0.0, 0.1, "Model")
#         axis_index+=1
# 
#         self._ax_2 = fig.add_subplot(num_plots_x, num_plots_y, axis_index, projection='3d')
#         self._ax_2.text2D(0.0, 0.1, "Entropy")
#         axis_index+=1
# 
#         #heat map for the model
#         self._ax_5 = fig.add_subplot(num_plots_x, num_plots_y, axis_index)
#         self._ax_5.axis([0, points_per_axis, 0, points_per_axis])
#         axis_index+=1
# 
#         #heat map for the acquisition function
#         self._ax_6 = fig.add_subplot(num_plots_x, num_plots_y, axis_index)
#         axis_index+=1
# 
#         self._ax_3 = fig.add_subplot(num_plots_x, num_plots_y, axis_index)
# #         self._ax_3.text2D(0.05, 1.95, "Candidates")
#         axis_index+=1
# 
#         self._ax_4 = fig.add_subplot(num_plots_x, num_plots_y, axis_index)
# #         self._ax_4.text2D(0.15, 1.95, "Pmin")
#         axis_index+=1
# 
#         #heat map for the cost gp
#         self._ax_7 = fig.add_subplot(num_plots_x, num_plots_y, axis_index)
#         axis_index+=1
# 
#         self._ax_8 = fig.add_subplot(num_plots_x, num_plots_y, axis_index)
#         self._ax_8.plot(rep[:, 0], rep[:, 1], 'b+')
#         self._ax_8.plot(rep[-1, 0], rep[-1, 1], 'ro')
# 
#         self._ax_3.axis([0, 1, 0, 1])
#         self._ax_4.axis([0, 1, 0, 1])
# 
#         plt.hold(True)
#         self._ax_3.set_xlabel('S')
#         self._ax_3.set_ylabel('X')
# 
#         self.plot3D_gp(model)
#         #add incumbent to plot (needs to be transposed)
#         self._ax_5.plot(incumbent[1] * points_per_axis, incumbent[0] * points_per_axis, 'mx')
# 
#         x = np.linspace(0, 1, points_per_axis)[:, np.newaxis]
#         y = np.linspace(0, 1, points_per_axis)[:, np.newaxis]
# 
#         x, y = np.meshgrid(x, y)
# 
#         test_inputs = np.zeros((points_per_axis * points_per_axis, 2))
# 
#         ei_values = np.zeros([points_per_axis, points_per_axis])
# 
#         for i in xrange(0, points_per_axis):
#             for j in xrange(0, points_per_axis):
#                 test_inputs[i * points_per_axis + j, 0] = x[j][i]
#                 test_inputs[i * points_per_axis + j, 1] = y[j][i]
#                 ei_values[j][i] = compute_expected_improvement(np.array([x[i][j], y[i][j]]), model)
# 
#         cost = cost_model.predict(test_inputs, False)
#         cost = cost.reshape([points_per_axis, points_per_axis])
#         self._ax_7.imshow(cost, cmap='hot', origin='lower')
#         self._ax_7.text(0, 105, "Costs")
#         self._ax_7.set_xlabel('X')
#         self._ax_7.set_ylabel('S')
# 
#         self.plot3D_entropy(cand[:, 0], cand[:, 1], entropy_values)
#         self._ax_6.plot(next_cand[1] * points_per_axis, next_cand[0] * points_per_axis, 'bx')
# 
#         self._ax_3.hold(True)
#         self._ax_3.plot(X[:, 0], X[:, 1], 'ro', label="Points")
#         self._ax_3.plot(cand[:, 0], cand[:, 1], 'b+',
#                                   label="Candidates")
#         self._ax_3.contour(x, y, ei_values, colors=('green'),
#                                   label="Expected Improvement")
# 
# 
#         #self._ax_3.legend()
# 
#         #self.plot3D_expected_improvement(x, y, ei_values)
# 
#         #self.plot3D_representer_points(entropy_estimator._func_sample_locations)
#         self._ax_4.hold(True)
#         self.plot_old_pmin(pmin)
# 
#         #self._ax_4.legend()
#         filename = self._path + "/plot3D_" + str(self._index) + ".png"
#         self._index += 1
# 
#         plt.savefig(filename)

    def plot3D_expected_improvement(self, x, y, ei_values):

        self._ax.plot_surface(x, y, ei_values, cmap='Greens_r')

    def plot3D_entropy(self, x, y, entropy):
        entropy = (entropy - np.mean(entropy)) / np.sqrt(np.var(entropy))
        self._ax_2.axis([0, 1, 0, 1])
#         self._ax_6.axis([0, self._points_per_axis, 0, self._points_per_axis])
        #self._ax_2.scatter(x, y, entropy, cmap='Blues_r')
        #self._ax_6.scatter(x, y, c=entropy, cmap='hot')
        _xi = np.linspace(0, 1, self._points_per_axis)
        _yi = np.linspace(0, 1, self._points_per_axis)
        _Z = mlab.griddata(x, y, entropy, _xi, _yi)
        _X, _Y = np.meshgrid(_xi, _yi)
        self._ax_2.plot_surface(_X, _Y, _Z.T, cmap='Blues_r')
#         self._ax_6.imshow(_Z.T, cmap='hot', origin='lower')
#         self._ax_6.text(0, self._points_per_axis + 5, "Acquisition Function")
#         self._ax_6.set_xlabel('X')
#         self._ax_6.set_ylabel('S')

    def plot3D_gp(self, model):

        x = np.linspace(0, 1, self._points_per_axis)[:, np.newaxis]
        y = np.linspace(0, 1, self._points_per_axis)[:, np.newaxis]

        x, y = np.meshgrid(x, y)
        test_inputs = np.zeros((self._points_per_axis * self._points_per_axis, 2))
        for i in xrange(0, self._points_per_axis):
            for j in xrange(0, self._points_per_axis):
                test_inputs[i * self._points_per_axis + j, 0] = x[j][i]
                test_inputs[i * self._points_per_axis + j, 1] = y[j][i]

        mean = model.predict(test_inputs, False)

        z = np.zeros((self._points_per_axis, self._points_per_axis))
        for i in xrange(0, self._points_per_axis):
            for j in xrange(0, self._points_per_axis):
                z[i][j] = mean[i * self._points_per_axis + j]

        self._ax.plot_surface(x, y, z, cmap='Reds_r')
#         self._ax_5.imshow(z, cmap='hot', origin='lower')
#         self._ax_5.text(0, self._points_per_axis + 5, "Model")
#         self._ax_5.set_xlabel('X')
#         self._ax_5.set_ylabel('S')

    def plot_new_pmin(self, pmin):

        ind = np.linspace(0, 1, pmin.shape[0])
        #self._ax_4.bar(ind, pmin, 0.04, color='b')
        self._ax_4.plot(ind, pmin, color='b', label="Pmin_new")

    def plot_old_pmin(self, pmin):

        ind = np.linspace(0, 1, pmin.shape[0])
        #self._ax_4.bar(ind, pmin, 0.04, color='r')
        self._ax_4.plot(ind, pmin, color='r', label="Pmin_old")


    #def plot_
