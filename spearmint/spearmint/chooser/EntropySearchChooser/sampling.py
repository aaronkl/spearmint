'''
Created on Aug 1, 2014

@author: Aaron Klein
'''

import emcee
import numpy as np


def sample_representer_points(starting_point, log_proposal_measurement, number_of_points, chain_length):

    nwalkers = number_of_points
    ndim = starting_point.shape[0]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_proposal_measurement)
    p0 = [starting_point for i in xrange(nwalkers)]
    pos, prob, state = sampler.run_mcmc(p0, 100)

    sampler.reset()

    sampler.run_mcmc(pos, chain_length, rstate0=state)

    #last points of the chains will be the representer points
    representer_points = sampler.chain[:, -1, :]

    return representer_points
