'''
Created on 01.04.2014

@author: Simon Bartels, Aaron Klein
'''
from abstract_test import AbstractTest, scale, d
import unittest
import numpy as np
import numpy.random as npr
from spearmint.chooser.EntropySearchChooser.entropy_search_chooser import EntropySearchChooser
from spearmint.chooser.EntropySearchChooser.hyper_parameter_sampling import sample_hyperparameters, sample_from_proposal_measure
from spearmint.chooser.EntropySearchChooser.gp_model import GPModel
import scipy.optimize as spo


class Test(AbstractTest):

    def setUp(self):
        super(Test, self).setUp()
        _hyper_samples = sample_hyperparameters(10, True, self.X, self.y,
                                                     self.cov_func, self.noise, self.amp2, self.ls)#[100-1]]
        self._models = []
        for h in range(0, len(_hyper_samples)):
            hyper = _hyper_samples[h]
            gp = GPModel(self.X, self.y, hyper[0], hyper[1], hyper[2], hyper[3])
            self._models.append(gp)

    def xtest_call_minimizer(self):
        '''
        Asserts that this function produces indeed something better than the starting point.
        '''
        opt_bounds = []# optimization bounds
        for i in xrange(0, d):
            opt_bounds.append((0, scale))

        chooser = EntropySearchChooser("")
        chooser._mcmc_iters = len(self._models)
        minima = chooser._find_local_minima(self.X, self.y, self._models, None)
        print minima

        #first dimension must be number of minima, second dimension the dimension of the points
        assert(minima.shape[1] == d)

        disturbed_minima = minima + npr.randn(minima.shape[0], d) / 1000
        disturbed_minima = np.clip(disturbed_minima, 0, 1)
        values = np.zeros(minima.shape[0])
        disturbed_values = np.zeros(minima.shape[0])
        gradients = np.zeros(minima.shape)
        for i in range(0, minima.shape[0]):
            values[i], gradients[i] = chooser._objective_function(minima[i], self._models, True)
            disturbed_values[i] = chooser._objective_function(disturbed_minima[i], self._models)

        #small perturbations of the minima should have higher values
        print values - disturbed_values
        assert(np.all(values - disturbed_values < 1e-15))

        print gradients
        # gradients should be almost 0 except at the borders
        assert(np.all(np.abs(gradients) < 1e-5))

    def test_minimizer_in_known_scenario(self):
        # TODO: for which reason ever the minimizer returns 1 as a minimum!
        # problematic seed: 5980
        n = 3
        X = np.array([np.linspace(1./(n+1), 1-1./(n+1), n)]).T
        y = -np.ones(n)
        chooser = EntropySearchChooser("")
        chooser._mcmc_iters = 1
        gp = GPModel(X, y, 0, 1e-15, 1, np.ones(1)/8)
        minima = chooser._find_local_minima(X, y, [gp], None)
        print minima
        for x in X:
            assert(np.any(np.abs(minima - x) < 1e-4))
        # print chooser._objective_function(np.array([0]), [gp], True)
        # print chooser._objective_function(np.array([1]), [gp], True)
        # print chooser._objective_function(np.array([0.5]), [gp], True)
        # starting_point =  np.array([1.])
        # opt_bounds = []# optimization bounds
        # for i in xrange(0, starting_point.shape[0]):
        #     opt_bounds.append((0, 1))
        # print spo.fmin_l_bfgs_b(chooser._objective_function, starting_point, args=([gp], True),
        #                                   bounds=opt_bounds, disp=0)

        # we have 3 minima and each should have almost same probability of being it
        assert(np.all(chooser._compute_pmin_probabilities([gp], minima) > 0.3))


    def test_gradient_computation(self):
        '''
        Asserts that the gradients are computed correctly.
        '''
        #problematic seed: 12233
        epsilon = 1e-6
        xstar = scale * npr.random(d)
        chooser = EntropySearchChooser("")
        chooser._mcmc_iters = len(self._models)
        #get gradient in the right shape (for the test)
        gradient = np.array([np.array([chooser._objective_function(xstar, self._models, True)[1]])])

        def f(x):
            return np.array([chooser._objective_function(x[0], self._models, False)])
        self.assert_first_order_gradient_approximation(f, np.array([xstar]), gradient, epsilon)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()