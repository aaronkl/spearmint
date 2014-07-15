'''
Created on 19.12.2013

@author: Simon Bartels

Provides common set up routine for tests.
'''
#from __future__ import absolute_import #turns off relative imports
import unittest
import spearmint.gp as gp
from spearmint.chooser.EntropySearchChooser.hyper_parameter_sampling import sample_hyperparameters
#import ..support.hyper_parameter_sampling as hps
from spearmint.chooser.EntropySearchChooser.gp_model import GPModel, getNumberOfParameters, fetchKernel
import numpy.random as npr
import numpy as np
import scipy.linalg as spla

'''
The number of input dimensions for the Gaussian Process.
'''
d = 3

'''
Scale/distribution of the inputs, i.e. factor to the uniform distribution.
'''
scale = 1 #25

def makeObservations(dimension, scale, ground_truth=None):
    N = npr.randint(2,25)
    def mod(x): return x % scale
    vec_mod = np.vectorize(mod)
    #uniformly distributed observations in [0,1) * scale
    X = vec_mod(scale * npr.random((N,dimension))) #% scale
    if not (ground_truth is None):
        y = ground_truth(X)
    else:
        y = npr.randn(N)
    return (X,y)

        
def copy_parameters(obj, gp):
    '''
    Copies the parameters of the GP into this object.
    Necessary for the spearmint GP.
    '''
    obj.cov_func = gp._cov_func
    obj.cov_grad_func = gp._covar_derivative
    obj.ls = gp._ls
    obj.noise = gp._noise
    obj.mean = gp._mean
    obj.amp2 = gp._amp2
    
def cov(gp, x1, x2=None):
    '''
    Spearmint covariance function.
    '''
    if x2 is None:
        return gp.amp2 * (gp.cov_func(gp.ls, x1, None)
                           + 1e-6*np.eye(x1.shape[0]))
    else:
        return gp.amp2 * gp.cov_func(gp.ls, x1, x2)

class AbstractTest(unittest.TestCase):
    def setUp(self):
        seed = npr.randint(65000)
        #seed = 35133
        print("using seed: " + str(seed))
        np.random.seed(seed)
        (X, y) = makeObservations(d, scale)
        self.X = X
        self.y = y
        covarname = "ARDSE"
        cov_func, _ = fetchKernel(covarname)
        noise = 1e-6
        amp2 = np.std(y)+1e-4
        ls = np.ones(getNumberOfParameters(covarname, d))
        noiseless = bool(npr.randint(2))
        parameter_ls = sample_hyperparameters(15, noiseless, X, y, cov_func, noise, amp2, ls)
        (mean, noise, amp2, ls) = parameter_ls[len(parameter_ls)-1]
        self.gp = GPModel(X, y, mean, noise, amp2, ls, covarname)
        copy_parameters(self, self.gp)

    def tearDown(self):
        pass
    
    def assert_first_order_gradient_approximation(self, f, x, dfdx, epsilon):
        '''
        Asserts that the computed gradient dfdx has the same sign and is about the same value as the
        first order approxmation.
        Args:
            f: the function
            x: the argument
            dfdx: the first order derivative of f in all arguments of x
            epsilon: the precision
        Returns:
            nothing, makes two assertions
        '''
        first_order_grad_approx = np.zeros([dfdx.shape[0], 1, d])
        for j in range(0,d):
            h = np.zeros(d)
            h[j] = epsilon
            first_order_grad_approx[:,0,j] = (f(x+h) - f(x-h))/(2*epsilon)
        #print "approximation: " + str(first_order_grad_approx[:,0]) + " computed: " + str(dfdx)
        #print dfdx.shape
        #print first_order_grad_approx.shape
        assertion_violations = 0
        wrong_sign = 0
        for i in range(0, dfdx.shape[0]):
            for j in range(0, d):
                signs_equal = np.sign(first_order_grad_approx[i,0,j]) == np.sign(dfdx[i,0,j])
                if not signs_equal and first_order_grad_approx[i,0,j] != 0:
                    print "gradient signs differ for element " + str(i) + " in dimension " + str(j) \
                    + "(" + str(first_order_grad_approx[i,0,j]) + " and " + str(dfdx[i,0,j]) + ")"
                    wrong_sign = wrong_sign + 1
                    continue
                dist = np.abs(first_order_grad_approx[i,0,j]-dfdx[i,0,j])
                precision = epsilon+0.1*np.abs(dfdx[i,0,j])
                if not dist < precision:
                    assertion_violations = assertion_violations + 1
                    print str(first_order_grad_approx[i,0,j]) + "-" + str(dfdx[i,0,j]) + " = " + str(dist) + " (> precision: " + str(precision) + ")"
                    print "(too many of these violations will cause the test to fail)"
                #assert(dist < precision)
        #assert(np.allclose(first_order_grad_approx, dfdx, rtol=0.01, atol=epsilon))
                        
        assert(wrong_sign < 3)
        assert(assertion_violations < 5)
    