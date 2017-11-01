
import numpy as np
from scipy.linalg import eigh as seigh
from GP.kernels import exponential_squared as expsq
import matplotlib.pyplot as plt



def p_from_lam(alpha, lam, w):
    p = np.zeros_like(lam)
    for i in xrange(lam.size):
        tmp = lam[i]**np.arange(alpha.size)
        p[i] = w[i]*np.exp(np.sum(alpha*tmp) - 1.0)
    return p

def get_err(plam, lam, mu):
    delmu = np.zeros_like(mu)
    for i in xrange(mu.size):
        delmu[i] = np.abs(np.sum(lam**i*plam) - mu[i])
    return np.max(delmu)

def MaxEntOpt(mu, lam, w, tol, maxit):
    M = mu.size #number of moments
    alpha = np.random.randn(mu.size) # initial guess for coeffs
    plam = p_from_lam(alpha, lam, w) # initial distribution

    # loop control
    i = 0  # counter over moments
    err = 1.0
    it = 0
    while err > tol and it < maxit:
        delta = np.log(mu[i]/np.sum(lam**i*plam))
        alpha[i] += delta
        plam = p_from_lam(alpha, lam, w)
        # get moment expectation
        err = get_err(plam, lam, mu)
        print delta, err
        i = (i+1)%M
        it += 1
    if it >= maxit:
        print "Max iterations reached"
    else:
        print "Converged on iteration %i"%it
    return alpha


if __name__=="__main__":
    N = 250 # number of data points
    # set hypers
    sigf = 0.25
    l = 1.0
    sign = 1.0
    theta = np.array([sigf, l, sign])
    # set domain
    t = np.linspace(1e-10,1,N)
    #get covariance matrix inclusing noise
    tt = np.tile(t, (N, 1)).T - np.tile(t, (N, 1))
    kernel = expsq.sqexp()
    K = kernel.cov_func(theta, tt, noise=True)
    # get true log-det
    L = np.linalg.cholesky(K)
    logdetK = 2.0*np.sum(np.log(np.diag(L)))

    # get max eigenvalue
    maxeig = seigh(K, eigvals_only=True, eigvals=(N-1,N-1))
    # normalise so we have a maximum possible eigenvalue of 1.0
    normfactr = 1.1*maxeig
    K /= normfactr

    # get true eigenvalues
    lam, V = np.linalg.eigh(K)
    #print (lam < 0.0).any()
    # estimate raw moments
    M = 30  # number of moments to use
    mu = np.zeros(M)
    for i in xrange(M):
        mu[i] = np.sum(np.diag(np.linalg.matrix_power(K, i)))/N  # exact in this case

    # get weights and abscissas
    x, w = np.polynomial.legendre.leggauss(21)
    I = np.argwhere(x >= 0)
    x = x[I]
    w = w[I]

    tol = 5e-5
    maxit = 10000
    alpha = MaxEntOpt(mu, x, w, tol, maxit)

    plam = p_from_lam(alpha, x, w)

    plt.figure('p')
    plt.plot(x, plam)
    plt.show()

    logdetK2 = N*np.sum(np.log(lam)*plam) + N*np.log(normfactr)

    print logdetK, logdetK2