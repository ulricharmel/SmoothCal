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
import Operators as ops

def interpolate(*args, **kwargs):
    """
    This here is to get the stack trace when an exception is thrown
    """
    try:
        return interpolate_impl(*args, **kwargs)
    except Exception as e:
        traceback_str = traceback.format_exc(e)
        raise StandardError("Error occurred. Original traceback "
                            "is\n%s\n" % traceback_str)


def interpolate_impl(theta, tp, gmean, gobs, K, Ky, D, k):
    """
    This is for the gain interpolation
    """
    gmean, gcov = Ky.interp(tp, theta, gobs, gmean)
    return gmean, gcov, k

def get_interp(theta, tp, gmean, gobs, K, Ky, D, Na):
    futures = []
    max_jobs = np.min(np.array([psutil.cpu_count(logical=False), Na]))
    gp = np.zeros([Na, tp.size], dtype=np.complex128)
    gpcov = np.zeros([Na, tp.size, tp.size], dtype=np.complex128)
    with cf.ProcessPoolExecutor(max_workers=max_jobs) as executor:
        for k in xrange(Na):
            future = executor.submit(interpolate, theta[k], tp, gmean[k], gobs[k], K[k], Ky[k], D[k], k)
            futures.append(future)
        for f in cf.as_completed(futures):
            gmean, gcov, k = f.result()
            gp[k] = gmean
            gpcov[k] = gcov
    return gp, gpcov


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

def update_impl(g0, gobs0, A, V, Sigma, K, Ky, D, i, k, lam=0.5):
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
    #gbar = (rhs_vec + g0) / 2.0
    gbar = (1.0-lam)* g0 + lam*rhs_vec
    #print (Ky.Sigmayinv*j).shape
    gobs = (Ky.Sigmayinv.dot(j) + gobs0)/2.0  # maximum likelihood solution for comparison
    return gbar, gobs, k

def get_update(g0, gobs0, A, V, Sigma, K, Ky, D, i, Na, lam=0.5):
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
            future = executor.submit(update, g0[k], gobs0[k], A[k], V[k], Sigma[k], K[k], Ky[k], D[k], i, k, lam)
            futures.append(future)
        for f in cf.as_completed(futures):
            g, gobs, k = f.result()
            g0[k] = g
            gobs0[k] = gobs
    return g0, gobs0

def stefcal_update(*args, **kwargs):
    """
    This here is to get the stack trace when an exception is thrown
    """
    try:
        return stefcal_update_impl(*args, **kwargs)
    except Exception as e:
        traceback_str = traceback.format_exc(e)
        raise StandardError("Error occurred. Original traceback "
                            "is\n%s\n" % traceback_str)

def stefcal_update_impl(g0, A, V, W, Sigmay, i, k):
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
    j = np.dot(A.T.conj(), W*V)
    # do update
    gbar = (j/Sigmay + g0)/2.0  # maximum likelihood solution for comparison
    return gbar, k

def get_stefcal_update(g0, A, V, W, JHJ, i, Na):
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
            future = executor.submit(stefcal_update, g0[k], A[k], V[k], W[k], JHJ[k], i, k)
            futures.append(future)
        for f in cf.as_completed(futures):
            g, k = f.result()
            g0[k] = g
    return g0

def train(*args, **kwargs):
    """
    This here is to get the stack trace when an exception is thrown
    """
    try:
        return train_impl(*args, **kwargs)
    except Exception as e:
        traceback_str = traceback.format_exc(e)
        raise StandardError("Error occurred. Original traceback "
                            "is\n%s\n" % traceback_str)

def train_impl(theta, j, Sigmay, W, V, K, Ky, D, g0):
    """
    Computes Hamiltonian and its gradient.
    Input:
            theta       - vector of hyperparameters with sigma_n at theta[-1]
            j           - information source A^H.Sigmainv.V
            Sigmay      - noise covariance of "observed" gains (vector od diagonals)
            W           - visibility weights
            K           - prior gain covariance matrix operator
            Ky          - observed gain covariance operator
            D           - posterior gain covariance operator
    We can probably gain quite a bit of efficiency using sparse arrays
    """
    Nt = g0.size
    N = V.size
    # update i and Sigmay
    j = j /theta[-1]**2
    Sigmay = Sigmay / theta[-1]**2
    # update hypers of prior covariance matrix
    K.update(theta)
    # compute Sigma
    Sigmainv = W/theta[-1]**2
    # update Ky and D
    Ky.update(K, Sigmay)
    D.update(K, Ky)
    # compute Hamiltonian
    logdetSigma = -np.sum(np.log(Sigmainv)) # negative because of inverse
    Kyinvg0 = Ky._idot(g0)
    # print logdetSigma
    # print K.logdet
    # print - D._logdet()
    # print (V.conj().T.dot(Sigmainv*V)).real
    # print -(j.conj().T.dot(D._dot(j))).real
    # print (g0.conj().T.dot(Kyinvg0)).real
    # print - 2*(j.conj().T.dot(g0) - j.conj().T.dot(K._dot(Kyinvg0))).real
    # print "pterm", (g0.conj().T.dot(Kyinvg0)).real - 2*(j.conj().T.dot(g0) - j.conj().T.dot(K._dot(Kyinvg0))).real
    H = logdetSigma + K.logdet - D._logdet() - (j.conj().T.dot(D._dot(j))).real + \
        (g0.conj().T.dot(Kyinvg0)).real - 2 * (j.conj().T.dot(g0) - j.conj().T.dot(K._dot(Kyinvg0))).real + (V.conj().T.dot(V * Sigmainv)).real

    # not operatorified below this line!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # get derivatives
    KyinvK = Ky._idot(K.val)
    dH = np.zeros(theta.size)
    for i in xrange(theta.size-1):
        #print i
        dKdtheta = K._dtheta(theta, mode=i)
        KyinvdKdtheta = Ky._idot(dKdtheta)
        dDdtheta = dKdtheta - dKdtheta.dot(KyinvK) + K._dot(KyinvdKdtheta.dot(KyinvK)) - K._dot(KyinvdKdtheta)
        # get the trace term
        term1 = np.einsum('ij,ji->i', K.KinvdKdtheta(theta, mode=i), KyinvK)
        term2 = -np.einsum('ij,ji->i', KyinvdKdtheta, KyinvK)
        term3 = np.diag(KyinvdKdtheta)
        term4 = -np.einsum('ij,ji->i', np.diag(Sigmay), dDdtheta)
        tterm = (np.sum(term1 + term2 + term3 + term4)).real
        #print "Trace", tterm
        # get data term
        dterm = (-j.conj().T.dot(dDdtheta.dot(j))).real
        #print "Data", dterm
        # get prior term
        KyinvdKdthetaKyinvg0 = KyinvdKdtheta.dot(Kyinvg0)
        pterm1 = (-g0.conj().T.dot(KyinvdKdthetaKyinvg0)).real
        pterm2 = 2*(j.conj().T.dot(dKdtheta.dot(Kyinvg0)) - j.conj().T.dot(K._dot(KyinvdKdthetaKyinvg0))).real
        #print "Prior", pterm1, pterm2
        dH[i] = tterm + dterm + pterm1 + pterm2
    # deriv w.r.t. sigma_n
    # get trace term
    i = theta.size - 1
    dKydsignKyinvK = Ky._dotdtheta(KyinvK, theta, i)  # check that dot is done correctly
    KyinvdKydsignKyinvK = Ky._idot(dKydsignKyinvK)
    dDdsign = K._dot(KyinvdKydsignKyinvK)
    term1 = (np.einsum('ij,ji->i', np.diag(Sigmay), dDdsign)).real
    term2 = (np.diag(KyinvdKydsignKyinvK)).real
    tterm = 2*N/theta[-1] - np.sum(term1 + term2)
    #print "Trace", tterm
    # get data term
    #print V.shape, Sigmainv.shape, (V/Sigma).shape
    dterm1 = (-2*V.conj().T.dot(V*Sigmainv)/theta[-1]).real
    dterm2 = (4*j.conj().T.dot(D._dot(j))/theta[-1]).real
    dterm3 = (- j.conj().T.dot(dDdsign.dot(j))).real
    dterm = dterm1 + dterm2 + dterm3
    #print "Data", dterm
    # get prior term
    pterm = 4*(j.conj().T.dot(g0) - j.conj().T.dot(K._dot(Kyinvg0))).real/theta[-1] - 2*(j.conj().T.dot(K._dot(Ky._idot(Ky._dotdtheta(Kyinvg0, theta, 1))))).real
    #print "Prior", pterm
    dH[i] = tterm + dterm + pterm
    print H, dH
    return H, dH

def get_hypers(theta, V, A, W, K, Ky, D, g0):
    futures = []
    H = 0.0
    Na = g0.shape[0]
    Nhypers = (theta.size - 1)/Na
    max_jobs = psutil.cpu_count(logical=False)
    with cf.ProcessPoolExecutor(max_workers=max_jobs) as executor:
        for i in xrange(Na):
            future = executor.submit(train, np.append(theta[i*Nhypers:(i+1)*Nhypers], theta[-1]), V[i], A[i], W[i], K[i], Ky[i], D[i], g0[i])
            futures.append(future)
        for f in cf.as_completed(futures):
            H += f.result()
    return H

def SmoothCal(Na, Nt, Xpq, Vpq, Wpq, t, theta0, tol=5e-3, maxiter=25, gbar=None, gobs=None):
    """
    The SmoothCal algorithm solves for the gains at times t
    Input:
        Na      - number of antennae
        Nt      - number of times
        Xpq     - model visibilities
        Vpq     - observed visbilities
        Wpq     - Visibility weights
        t       - times at which to solve for gains
        theta0  - initial guess for hyperparameters
        tol     - convergence criterion, stop when maximum difference between successive updates is less than tol
        maxiter - maximum number of iterations
    """
    print "Running SmoothCal cycle"
    # check shapes
    assert t.size == Nt
    assert Vpq.shape == (Na, Na, Nt)
    assert Xpq.shape == (Na, Na, Nt)

    Nhypers = theta0.size

    # set operators and data structures for doing per antennae solve
    A = np.zeros([Na, Nt*Na, Nt], dtype=np.complex128) # to hold per-antenna response
    V = np.zeros([Na, Nt*Na], dtype=np.complex128) # to hold per-antenna data
    W = np.ones([Na, Nt*Na], dtype=np.float64) # to hold weights
    Sigma = np.zeros([Na, Na*Nt], dtype=np.complex128) # to hold per-antenna weights
    Sigmay = np.zeros([Na, Nt], dtype=np.complex128) # to hold diagonal of Ad.Sigmainv.A
    theta = np.zeros([Na, Nhypers]) # hyper-parameters excluding sigman
    Klist = []  # list holding covariance operators
    Kylist = [] # list holding Ky operators
    Dlist = []  # list holding D operators

    # initial guess for gains
    if gbar is None:
        gbar = np.ones([Na, Nt], dtype=np.complex) # initial guess for posterior mean
    if gobs is None:
        gobs = np.ones([Na, Nt], dtype=np.complex)  # initial guess for maximum likelihood solution

    # start iterations
    lam=0.5
    diff = 1.0
    i = 0
    while diff > tol and i < maxiter:
        gold = gbar.copy()
        gobsold = gobs.copy()
        for p in xrange(Na):
            for j in xrange(Nt):
                Rpt = Xpq[p, :, j]*(gold[:, j].conj())
                #Robspt = Xpq[p, :, j] * (gobsold[:, j].conj())
                #A[p*Na*Nt + j*Na:p*Na*Nt + j*Na + Na, p*Nt + j] = Rpt
                A[p, j*Na:(j+1)*Na, j] = Rpt
                #Aobs[p, j * Na:(j + 1) * Na, j] = Robspt
                if i == 0:
                    #V[p*Na*Nt + j*Na:p*Na*Nt + j*Na + Na] = Vpq[p,:,j]  # need to change this if autocor is False
                    V[p, j*Na:(j+1)*Na] = Vpq[p, :, j]
                    W[p, j*Na:(j+1)*Na] = Wpq[p, :, j]
            if i==0:
                Sigma[p] = theta0[-1]**2/W[p]
                Sigmay[p] = np.diag(np.dot(A[p].T.conj(), np.diag(1.0/Sigma[p]).dot(A[p])))
                theta[p] = theta0 #+ deltheta*np.random.randn(Nhypers))  # want common sigman
                Klist.append(ops.K_operator(t, theta0))
                Kylist.append(ops.Ky_operator(Klist[p], Sigmay[p], solve_mode="full"))
                Dlist.append(ops.D_operator(Klist[p], Kylist[p]))
            else:
                Sigmay[p] = np.diag(np.dot(A[p].T.conj(), np.diag(1.0/Sigma[p]).dot(A[p])))
                Kylist[p].update(Klist[p], Sigmay[p])
                Dlist[p].update(Klist[p], Kylist[p])

        # Solve for mean
        diffold = diff
        gbar, gobs = get_update(gold.copy(), gobsold.copy(), A, V, Sigma, Klist, Kylist, Dlist, i, Na, lam=lam)
        diff = np.max(np.abs(gbar-gold))
        if diff >= diffold:
            lam = 0.5*lam


        i += 1
        print "At iteration %i maximum difference is %f"%(i, diff)

    if i >= maxiter:
        print "Maximum iterations reached"

    return gbar, gobs, Klist, Kylist, Dlist, theta


def StefCal(Na, Nt, Xpq, Vpq, Wpq, t, tol=5e-3, maxiter=25):
    """
    The SmoothCal algorithm solves for the gains at times t
    Input:
        Na      - number of antennae
        Nt      - number of times
        Xpq     - model visibilities
        Vpq     - observed visbilities
        Wpq     - Visibility weights
        t       - times at which to solve for gains
        theta0  - initial guess for hyperparameters
        tol     - convergence criterion, stop when maximum difference between successive updates is less than tol
        maxiter - maximum number of iterations
    """
    print "Running StefCal cycle"
    # check shapes
    assert t.size == Nt
    assert Vpq.shape == (Na, Na, Nt)
    assert Xpq.shape == (Na, Na, Nt)


    # set operators and data structures for doing per antennae solve
    A = np.zeros([Na, Nt*Na, Nt], dtype=np.complex128) # to hold per-antenna response
    V = np.zeros([Na, Nt*Na], dtype=np.complex128) # to hold per-antenna data
    W = np.ones([Na, Nt*Na], dtype=np.float64) # to hold weights
    JHJ = np.zeros([Na, Nt], dtype=np.float64) # to hold diagonal of Ad.Sigmainv.A

    # initial guess for gains
    gbar = np.ones([Na, Nt], dtype=np.complex128) # initial guess for posterior mean

    # start iterations
    diff = 1.0
    i = 0
    while diff > tol and i < maxiter:
        gold = gbar.copy()
        for p in xrange(Na):
            for j in xrange(Nt):
                Rpt = Xpq[p, :, j]*(gold[:, j].conj())
                A[p, j*Na:(j+1)*Na, j] = Rpt
                if i == 0:
                    V[p, j*Na:(j+1)*Na] = Vpq[p, :, j]
                    W[p, j*Na:(j+1)*Na] = Wpq[p, :, j]

            JHJ[p] = np.diag(np.dot(A[p].T.conj(), np.diag(W[p]).dot(A[p]))).real

        # Maximum likelihood solution
        gbar = get_stefcal_update(gold.copy(), A, V, W, JHJ, i, Na)
        diff = np.max(np.abs(gbar-gold))

        i += 1
        print "At iteration %i maximum difference is %f"%(i, diff)

    if i >= maxiter:
        print "Maximum iterations reached"

    return gbar, JHJ


def Stefcal_over_time(Na, Nt, Xpq, Vpq, Wpq, t, tol=5e-3, maxiter=25, t_int=1):
    """
    Averages over Nt_interval time samples 
    :param Na: 
    :param Nt: 
    :param Xpq: 
    :param Vpq: 
    :param Wpq: 
    :param Nt_interval: 
    :return: 
    """

    print "Running StefCal cycle with solution interval %d"%t_int
    # check shapes
    assert t.size == Nt
    assert Vpq.shape == (Na, Na, Nt)
    assert Xpq.shape == (Na, Na, Nt)

    Nbin = int(np.ceil(float(Nt))/t_int)
    
    # set operators and data structures for doing per antennae solve
    A = np.zeros([Na, Nt*Na, Nbin], dtype=np.complex128) # to hold per-antenna response
    V = np.zeros([Na, Nt*Na], dtype=np.complex128) # to hold per-antenna data
    W = np.ones([Na, Nt*Na], dtype=np.float64) # to hold weights
    JHJ = np.zeros([Na, Nbin], dtype=np.float64) # to hold diagonal of Ad.Sigmainv.A

    # initial guess for gains
    gbar = np.ones([Na, Nbin], dtype=np.complex128) # initial guess for posterior mean

    # start iterations
    diff = 1.0
    i = 0
    while diff > tol and i < maxiter:
        A = np.zeros([Na, Nt*Na, Nbin], dtype=np.complex128) # to hold per-antenna response
        
        gold = gbar.copy()
        for p in xrange(Na):
            for j in xrange(Nt):
                rr = j/t_int
                Rpt = Xpq[p, :, j]*(gold[:, rr].conj())
                A[p, j*Na:(j+1)*Na, rr] += Rpt
                if i == 0:
                    V[p, j*Na:(j+1)*Na] = Vpq[p, :, j]
                    W[p, j*Na:(j+1)*Na] = Wpq[p, :, j]

            JHJ[p] = np.diag(np.dot(A[p].T.conj(), np.diag(W[p]).dot(A[p]))).real

            # Maximum likelihood solution
        gbar = get_stefcal_update(gold.copy(), A, V, W, JHJ, i, Na)
        diff = np.max(np.abs(gbar-gold))

        i += 1
        print "At iteration %i maximum difference is %f"%(i, diff)

    if i >= maxiter:
        print "Maximum iterations reached"

    return gbar, JHJ


def Hogbom(ID, PSF, gamma=0.1, peak_fact=0.1, maxiter=10000):
    peak_flux = np.abs(ID.max())
    stopping_flux = peak_flux*peak_fact
    print "Peak flux = ", peak_flux, "Stopping flux = ", stopping_flux
    IM = np.zeros_like(ID)
    IR = ID.copy()
    Npix = IM.shape[0]
    j = 0
    while j < maxiter and peak_flux > stopping_flux:
        # get the position of the peak
        p, q = np.argwhere(np.abs(IR) == peak_flux).squeeze()
        IM[p, q] += gamma*peak_flux
        Istar = IR[p, q]
        IR -= gamma*Istar*PSF[Npix-1 - p:(2*Npix-1) - p, Npix-1 - q:(2*Npix-1) - q]
        peak_flux = np.abs(IR).max()
        j += 1
    if j >= maxiter:
        print "Maximum iterations reached"

    print "Stopping flux = ", peak_flux
    return IM, IR


# def Average_over_time(Na, Nt, Xpq, Vpq, Wpq, Nbin, gpred, t):
#     """
#     Averages over Nt_interval time samples 
#     :param Na: 
#     :param Nt: 
#     :param Xpq: 
#     :param Vpq: 
#     :param Wpq: 
#     :param Nt_interval: 
#     :return: 
#     """
#     # get new number of time bins
#     Nt_new = Nt//Nbin
#     # find new bin centers
#     t_new = np.linspace(t[0], t[-1], Nt_new)
#     # set up storage arrays
#     A = np.zeros([Na, Nt_new*Na, Nt_new], dtype=np.complex128) # to hold per-antenna response
#     V = np.zeros([Na, Nt_new*Na], dtype=np.complex128) # to hold per-antenna data
#     W = np.ones([Na, Nt_new*Na], dtype=np.float64) # to hold weights
#     Sigma = np.zeros([Na, Na*Nt_new], dtype=np.complex128) # to hold per-antenna weights
#     Sigmay = np.zeros([Na, Nt_new], dtype=np.complex128) # to hold diagonal of Ad.Sigmainv.A
#     for p in xrange(Na):
#         for j in xrange(Nt_new):
#             # get averaged quantities
#             tmp = Xpq[p, :, j*Nbin:(j+1)*Nbin] * gpred[:, j*Nbin:(j+1)*Nbin].conj()
#             print tmp.shape
#             Rpt = np.mean(tmp, axis=1)
#             A[p, j * Na:(j + 1) * Na, j] = Rpt
#             V[p, j * Na:(j + 1) * Na] = np.mean(Vpq[p, :, j*Nbin:(j+1)*Nbin], axis=1)
#             W[p, j * Na:(j + 1) * Na] = np.mean(Wpq[p, :, j*Nbin:(j+1)*Nbin], axis=1)
#         Sigma[p] = 1.0 / W[p]
#         Sigmay[p] = np.diag(np.dot(A[p].T.conj(), np.diag(1.0 / Sigma[p]).dot(A[p])))
#     return A, V, Sigmay, t_new, Nt_new