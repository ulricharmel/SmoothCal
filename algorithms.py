#!/usr/bin/env python

"""
Created on Wed Oct 18 18:52:40 2017

@author: landman

Here I define all the algorithms required for implementing SmoothCal.


"""

import numpy as np
import traceback
import concurrent.futures as cf
import psutil
from scipy.integrate import quad
from scipy.linalg import eigh as seigh
from GP.kernels import exponential_squared as expsq
import matplotlib.pyplot as plt

def update(*args, **kwargs):
    """
    This here is to get the stack trace when an exception is thrown
    """
    try:
        return update_impl(*args, **kwargs)
    except Exception as e:
        traceback_str = traceback.format_exc(e)
        raise StandardError("Error occurred. Original traceback "
                            "is\n%s\n" % traceback_str)

def update_impl(g0, gobs0, A, V, Sigma, K, Ky, D, i, k):
    """
    Here we compute the update for a single antenna.
    Input:
        g0      - the previous value of the gain
        A       - the response for the antenna
        V       - data vector for the antenna
        Sigma   - noise covariance as a vector
        K       - Linear operator representation of prior covariance matrix for antenna
        Ky      - Linear operator representation of Ky
        D       - Linear operator representation of antenna covariance matrix
        i       - the iteration number
        k       - process number to keep everything in the same order
    """
    # compute data source i.e. j
    j = np.dot(A.T.conj(), V/Sigma)
    rhs_vec = K._dot(j) + g0
    rhs_vec = rhs_vec - K._dot(Ky._idot(rhs_vec))
    gbar = (rhs_vec + g0) / 2.0
    #print (Ky.Sigmayinv*j).shape
    gobs = (Ky.Sigmayinv.dot(j) + gobs0)/2.0  # maximum likelihood solution for comparison
    return gbar, gobs, k

def get_update(g0, gobs0, A, V, Sigma, K, Ky, D, i, Na):
    """
    Here we compute the update for a single antenna. 
    Input:
        g0      - the previous value of the gain
        A       - the response for all antennae
        V       - data vector for all antennae
        Sigma   - noise covariance as a vector
        K       - List of linear operators holding prior covariance matrices for all antennae 
        Ky      - List of linear operators holding Ky
        D       - List of linear operators holding antennae covariance matrices
        i       - the iteration number
    """
    futures = []
    max_jobs = np.min(np.array([psutil.cpu_count(logical=False), Na]))
    with cf.ProcessPoolExecutor(max_workers=max_jobs) as executor:
        for k in xrange(Na):
            future = executor.submit(update, g0[k], gobs0[k], A[k], V[k], Sigma[k], K[k], Ky[k], D[k], i, k)
            futures.append(future)
        for f in cf.as_completed(futures):
            g, gobs, k = f.result()
            g0[k] = g
            gobs0[k] = gobs
    return g0, gobs0




