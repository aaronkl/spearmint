'''
Created on 01.04.2014

@author: Aaron Klein
'''
import unittest
import numpy as np
import scipy.linalg as spla
import scipy.stats as sps

from spearmint.chooser.EntropySearchChooser.entropy import Entropy
from spearmint.chooser.EntropySearchChooser.gp_model import GPModel, fetchKernel, getNumberOfParameters
from spearmint.chooser.EntropySearchChooser.entropy_with_costs import EntropyWithCosts
from spearmint.chooser.EntropySearchChooser.support import compute_expected_improvement

from spearmint.chooser.OptSizeChooser.acquisition_functions.entropy_search import EntropySearch
from spearmint.chooser.OptSizeChooser.acquisition_functions.entropy_search_big_data import EntropySearchBigData



class Test(unittest.TestCase):

    def setUp(self):

        self._seed = np.random.randint(65000)
        print("using seed: " + str(self._seed))

        dim = 1
        N = 3
        self._X = np.zeros([N, dim])
        self._y = np.zeros(N)
        self._X[0][0] = 0.2
        self._y[0] = 0
        self._X[1][0] = 0.8
        self._y[1] = 0
        self._X[2][0] = 0.0
        self._y[2] = 0
        self.covarname = "Matern52"
        self.mean = 0
        self.noise = 1e-6
        self.amp2 = 1
        
        self.num_of_hal_vals = 300
        self.num_of_samples = 300
        self.num_of_rep_points = 10
        self.chain_length = 20
        
        self.cov_func, _ = fetchKernel(self.covarname)
        
        self.ls = np.ones(getNumberOfParameters(self.covarname, dim)) / 4
        self._gp = GPModel(self._X, self._y, self.mean, self.noise, self.amp2, self.ls, self.covarname)

        self._durs = np.zeros(N)
        self._durs[0] = 0.5
        self._durs[1] = 1
        self._durs[2] = 1
        self._cost_gp = GPModel(self._X, self._y, 10, self.noise, self.amp2, self.ls, self.covarname)

        self._candidate = np.random.uniform(0, 1, self._X.shape[1])

    def test_compute_entropy(self):

        np.random.seed(self._seed)
        es = EntropySearch(self._X, self._y, self._gp)

        np.random.seed(self._seed)
        entropy = Entropy(self._gp, self.num_of_hal_vals, self.num_of_samples, self.num_of_rep_points, self.chain_length)
        
        self.assertEqual(entropy.compute(self._candidate), es.compute_fast(self._candidate))

        pass
    
    def test_compute_entropy_with_costs(self):
        
        np.random.seed(self._seed)
        es = EntropySearchBigData(self._X, self._y, self._gp, self._cost_gp)
        
        np.random.seed(self._seed)
        entropy = EntropyWithCosts(self._gp, self.num_of_hal_vals, self.num_of_samples, self.num_of_rep_points, self.chain_length)
        
        self.assertEqual(entropy._kl_divergence_old, es._kl_divergence_old)
        
        self.assertEqual(entropy.compute(self._candidate), es.compute(self._candidate))
        
        pass
    
    def xtest_compute_pmin(self):
                
        np.random.seed(self._seed)
        es = EntropySearch(self._X, self._y, self._gp)

        np.random.seed(self._seed)
        entropy = Entropy(self._gp, self.num_of_hal_vals, self.num_of_samples, self.num_of_rep_points, self.chain_length)
        
        mean, L = self._gp.getCholeskyForJointSample(np.append(np.array([self._candidate]),
                                                        entropy._representer_points,
                                                        axis=0))

        l = np.copy(L[1:, 0])
        mean = np.copy(mean[1:])
        L = np.copy(L[1:, 1:])
                
        assert(np.all(np.abs(entropy._compute_pmin(mean, L) - es._compute_pmin_bins_faster(mean, L)) < 1e-50))
                
        pass
    
    def test_compute_expected_improvement(self):
        #Spearmint code that computes ei
        vals = self._y
        comp = self._X
        cand = np.array([self._candidate])
    
        best = np.min(vals)

        comp_cov   = self.cov(comp)
        cand_cross = self.cov(comp, cand)

        obsv_cov  = comp_cov + self.noise*np.eye(comp.shape[0])
        obsv_chol = spla.cholesky( obsv_cov, lower=True )

        alpha  = spla.cho_solve((obsv_chol, True), vals - self.mean)
        beta   = spla.solve_triangular(obsv_chol, cand_cross, lower=True)

        func_m = np.dot(cand_cross.T, alpha) + self.mean
        func_v = self.amp2*(1+1e-6) - np.sum(beta**2, axis=0)

        func_s = np.sqrt(func_v)
        u      = (best - func_m) / func_s
        ncdf   = sps.norm.cdf(u)
        npdf   = sps.norm.pdf(u)
        ei     = func_s*( u*ncdf + npdf)
            
        self.assertEqual(compute_expected_improvement(self._candidate, self._gp), ei)
                
        pass
    
    def test_log_proposal_measurement(self):
        
        np.random.seed(self._seed)
        es = EntropySearch(self._X, self._y, self._gp)

        np.random.seed(self._seed)
        entropy = Entropy(self._gp)
        
        self.assertEqual(entropy._log_proposal_measure(self._candidate), es._log_proposal_measure(self._candidate))
        
        pass

    def test_sample_measure(self):
         
        x = np.ones(self._candidate.shape[0] - 1)
        np.random.seed(self._seed)
        es = EntropySearchBigData(self._X, self._y, self._gp, self._cost_gp)
 
        np.random.seed(self._seed)
        entropy = EntropyWithCosts(self._gp, self._cost_gp, self.num_of_hal_vals, self.num_of_samples, self.num_of_rep_points)
         
        self.assertEqual(entropy._sample_measure(x), es._sample_measure(x))
         
        pass
        
    #Spearmint
    def cov(self, x1, x2=None):
        if x2 is None:
            return self.amp2 * (self.cov_func(self.ls, x1, None)
                               + 1e-6*np.eye(x1.shape[0]))
        else:
            return self.amp2 * self.cov_func(self.ls, x1, x2)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
