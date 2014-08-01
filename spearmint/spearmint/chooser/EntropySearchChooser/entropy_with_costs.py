'''
Created on 02.12.2013

@author: Aaron Klein, Simon Bartels

This class extends EntropySearch to be used for big data sets. I.e. it incorporates the cost models
and it ASSUMES that the first value of a point is the data set size.
'''

from sampling import sample_representer_points

import numpy as np
from .support import compute_expected_improvement, compute_kl_divergence, sample_from_proposal_measure
from scipy.stats import norm
from sobol_lib import i4_sobol_generate
from util import slice_sample
#from .hyper_parameter_sampling import sample_from_proposal_measure


class EntropyWithCosts():

    def __init__(self, gp, cost_gp, num_of_hal_vals=21, num_of_samples=500, num_of_rep_points=20, chain_length=20,
                 transformation=8):

        self._gp = gp
        self._cost_gp = cost_gp
        self._num_of_hallucinated_vals = num_of_hal_vals
        self._num_of_samples = num_of_samples
        self._num_of_representer_points = num_of_rep_points
        if transformation not in range(1, 11):
            raise NotImplementedError("There exists no transformation with number " + str(transformation))
        self._transformation = transformation

        comp = gp.getPoints()
        vals = gp.getValues()

        points = i4_sobol_generate(self._gp.getPoints().shape[1], 100, 1)
        points[0, :] = 1

        #Evaluate EI of a sobel gird to find a good starting point to sample the representer points
        ei_vals = np.zeros([points.shape[1]])
        for i in xrange(0, points.shape[1]):
            ei_vals[i] = compute_expected_improvement(points[:, i], self._gp)

        idx = np.argmax(ei_vals)
        starting_point = points[:, idx]
        starting_point = starting_point[1:]

        #representers = sample_from_proposal_measure(starting_point, self._sample_measure, num_of_rep_points - 1, chain_length)
        representers = sample_representer_points(starting_point, self._sample_measure, num_of_rep_points - 1, chain_length)

        self._representer_points = np.empty([num_of_rep_points, comp.shape[1]])
        self._log_proposal_vals = np.zeros(num_of_rep_points)

        for i in range(0, num_of_rep_points - 1):
            self._log_proposal_vals[i] = self._sample_measure(representers[i])
            #set first value to one
            self._representer_points[i] = np.insert(representers[i], 0, 1)

        incumbent = comp[np.argmin(vals)][1:]
        self._representer_points[-1] = np.insert(incumbent, 0, 1)
        self._log_proposal_vals[-1] = self._sample_measure(incumbent)

        #as fortran array Omega.T is C-contiguous which speeds up dot product computation
        self._Omega = np.asfortranarray(np.random.normal(0, 1, (self._num_of_samples,
                                             self._num_of_representer_points)))

        self._hallucinated_vals = norm.ppf(np.linspace(1. / (self._num_of_hallucinated_vals + 1),
                                           1 - 1. / (self._num_of_hallucinated_vals + 1),
                                           self._num_of_hallucinated_vals))

        #TODO: Change it to vector computation
        self._pmin_old = self._compute_pmin_old(self._gp)

        entropy_pmin_old = -np.dot(self._pmin_old, np.log(self._pmin_old + 1e-50))

        log_proposal_old = np.dot(self._log_proposal_vals, self._pmin_old)
        self._kl_divergence_old = -(entropy_pmin_old - log_proposal_old)

        self._idx = np.arange(0, self._num_of_samples)

    def compute(self, candidate):

        kl_divergence = -compute_kl_divergence(candidate, self._representer_points, self._log_proposal_vals,
                                              self._gp, self._Omega, self._hallucinated_vals)
        scale = self._cost_gp.predict(np.array([candidate]))
        return self._transform(kl_divergence, scale)

    def _transform(self, kl_divergence, scale):
        '''
        Transforms the expected Kullback-Leibler divergence and the predicted cost.
        Args:
            kl_divergence: log of the expected Kullback-Leibler divergence
            scale: predicted costs to evaluate the candidate
        Returns:
            a scalar
        '''
        scale = np.max([1e1, scale])
        if self._transformation == 1:
            if kl_divergence < 0:
                return kl_divergence * scale
            return kl_divergence / scale
        elif self._transformation == 2:
            return np.exp(kl_divergence) / scale
        elif self._transformation == 3:
            if kl_divergence < 0:
                return kl_divergence * np.log(scale)
            return kl_divergence / np.log(scale)
        elif self._transformation == 4:
            return np.exp(kl_divergence) / np.log(scale)
        elif self._transformation == 5:
            return (np.exp(kl_divergence) - np.exp(self._kl_divergence_old)) / np.log(scale)
        elif self._transformation == 6:
            return np.exp(kl_divergence - self._kl_divergence_old) / np.log(scale)
        elif self._transformation == 7:
            return (kl_divergence - self._kl_divergence_old) / np.log(scale)
        elif self._transformation == 8:
            return (kl_divergence - self._kl_divergence_old) / scale
        elif self._transformation == 9:
            return (np.exp(kl_divergence) - np.exp(self._kl_divergence_old)) / scale
        elif self._transformation == 10:
            return np.exp(kl_divergence - self._kl_divergence_old) / scale

    def _sample_measure(self, x):

        if np.any(x < 0) or np.any(x > 1):
            return -np.inf

        #set first value to one
        x = np.insert(np.array(x), 0, 1)
        v = compute_expected_improvement(x, self._gp)
        return np.log(v + 1e-10)

    def _compute_pmin_old(self, gp):

        pmin = np.zeros(self._num_of_representer_points)
        mean, L = gp.getCholeskyForJointSample(self._representer_points)

        for omega in self._Omega:
            vals = gp.drawJointSample(mean, L, omega)
            mins = np.where(vals == vals.min())[0]
            number_of_mins = len(mins)
            for m in mins:
                pmin[m] += 1. / (number_of_mins)

        pmin = pmin / self._num_of_samples

        return pmin
