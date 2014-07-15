'''
Created on 02.12.2013

@author: Aaron Klein, Simon Bartels

'''
import numpy as np
from util import slice_sample
from .support import compute_expected_improvement, sample_from_proposal_measure, compute_kl_divergence
from scipy.stats import norm
from sobol_lib import i4_sobol_generate


class Entropy(object):

    def __init__(self, gp, num_of_hal_vals=21, num_of_samples=500, num_of_rep_points=20, chain_length=20):

        #Number of samples for the current candidate
        self._num_of_hallucinated_vals = num_of_hal_vals
        #Number of function drawn from the gp
        self._num_of_samples = num_of_samples
        #Number of point where pmin is evaluated
        self._num_of_representer_points = num_of_rep_points
        self._gp = gp
        self._idx = np.arange(0, self._num_of_samples)

        #as fortran array Omega.T is C-contiguous which speeds up dot computation
        self._Omega = np.asfortranarray(np.random.normal(0, 1, (self._num_of_samples,
                                             self._num_of_representer_points)))

        self._hallucinated_vals = norm.ppf(np.linspace(1. / (self._num_of_hallucinated_vals + 1),
                                    1 - 1. / (self._num_of_hallucinated_vals + 1),
                                    self._num_of_hallucinated_vals))

        comp = gp.getPoints()
        vals = gp.getValues()
        incumbent = comp[np.argmin(vals)]

        points = i4_sobol_generate(self._gp.getPoints().shape[1], 100, 1)

        #Evaluate EI of a sobel gird to find a good starting point to sample the representer points
        ei_vals = np.zeros([points.shape[1]])
        for i in xrange(0, points.shape[1]):
            ei_vals[i] = compute_expected_improvement(points[:, i], self._gp)

        idx = np.argmax(ei_vals)
        start_point = points[:, idx]

        self._representer_points = sample_from_proposal_measure(start_point, self._log_proposal_measure,
                                                                num_of_rep_points - 1, chain_length)

        #Add the incumbent to the representer points
        self._representer_points = np.vstack((self._representer_points, incumbent))

        self._log_proposal_vals = np.zeros(self._num_of_representer_points)

        for i in range(0, self._num_of_representer_points):
            self._log_proposal_vals[i] = self._log_proposal_measure(self._representer_points[i])

    def _log_proposal_measure(self, x):

        if np.any(x < 0) or np.any(x > 1):
            return -np.inf

        v = compute_expected_improvement(x, self._gp)

        return np.log(v + 1e-10)

    def compute(self, candidate):

        return -compute_kl_divergence(candidate, self._representer_points, self._log_proposal_vals,
                                      self._gp, self._Omega, self._hallucinated_vals)
