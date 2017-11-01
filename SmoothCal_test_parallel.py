"""
Created on Tue Oct 17 17:54:03 2017

@author: landman

Testing parallel version

"""

#import mkl
#mkl.set_num_threads(1)
import numpy as np
import numpy.ma as ma
import traceback
import psutil
import itertools as it
from scipy import optimize as opt
import concurrent.futures as cf
import matplotlib.pyplot as plt
import Operators as ops
import algorithms as algos
from GP.tools import draw_samples
#from pyrap.tables import table as pt

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

    #H2 = get_full_decomp(theta, V, A, t, g0.flatten(), Nt, Na)
    print H
    return H




if __name__=="__main__":
    Nfull = 550
    tfull = np.linspace(-5.5, 5.5, Nfull)
    # set time domain
    interval = 50
    Nt = 300
    t = tfull[0:interval]
    tp = tfull[interval:2*interval]
    for i in xrange(2,Nfull//interval):
        if i%2==0:
            t = np.append(t, tfull[i*interval:(i+1)*interval])
        else:
            tp = np.append(tp, tfull[i * interval:(i + 1) * interval])
    # t = np.sort(-5.5 + 11*np.random.random(Nt))
    # Ntp = 500
    # tp = np.linspace(-5.5, 5.5, Ntp)

    # number of antennae
    Na = 3

    # set mean functions for amplitude and phase
    meanfr = np.ones(Nfull, dtype=np.float64)
    meanfi = np.zeros(Nfull, dtype=np.float64)

    # set covariance params
    lGP = 0.5
    sigmaf = 0.25
    sigman = 0.2
    theta0 = np.array([sigmaf, lGP])
    Nhypers = theta0.size

    # sample gains
    def cov_func(x, theta):
        return theta[0]**2*np.exp(-x**2/(2*theta[1]**2))
    gfull = draw_samples.draw_samples(meanfr, tfull, theta0, cov_func, Na) + 1.0j*draw_samples.draw_samples(meanfi, tfull, theta0, cov_func, Na)

    g = np.zeros([Na, Nt], dtype=np.complex128)
    for i in xrange(0,Nfull//interval - Nfull//(2*interval)):
        g[:, i*interval:(i+1)*interval] = gfull[:, 2*i*interval:(2*i+1)*interval]

    # make sky model
    Npix = 33
    l = np.linspace(-1.0, 1.0, Npix)
    m = np.linspace(-1.0, 1.0, Npix)
    ll, mm = np.meshgrid(l, m)
    lm = (np.vstack((ll.flatten(), mm.flatten())))
    IM = np.zeros([Npix, Npix])
    IM[Npix//2, Npix//2] = 10.0
    IM[Npix//4, Npix//4] = 1.0
    IM[3*Npix//4, 3*Npix//4] = 1.0
    IM[Npix//4, 3*Npix//4] = 1.0
    IM[3*Npix//4, Npix//4] = 1.0
    IMflat = IM.flatten()

    # this is to create the pq iterator
    tmp = '1'
    for i in xrange(2, Na+1):
        tmp += str(i)

    # iterator over antenna pairs
    autocor = True
    if autocor:
        pqlist = list(it.combinations_with_replacement(tmp,2))
        N = Na*(Na+1)//2 #number of antenna pairs including autocor
    else:
        pqlist = list(it.combinations(tmp,2))
        N = Na*(Na-1)//2 #number of antenna pairs excluding autocor

    # choose random antennae locations
    u = np.random.random(Na)
    v = np.random.random(Na)

    # create baselines with time axis
    upq = np.zeros([N, Nt])
    vpq = np.zeros([N, Nt])
    phi = np.linspace(0, np.pi, Nt) # to simulate earth rotation
    for i, pq in enumerate(iter(pqlist)):
        upq[i, 0] = u[int(pq[0])-1] - u[int(pq[1])-1]
        vpq[i, 0] = v[int(pq[0])-1] - v[int(pq[1])-1]
        for j in xrange(1, Nt):
            rottheta = np.array([[np.cos(phi[j]), -np.sin(phi[j])], [np.sin(phi[j]), np.cos(phi[j])]])
            upq[i, j], vpq[i, j] = np.dot(rottheta, np.array([upq[i, 0], vpq[i, 0]]))

#    # inspect uv-coverage
#    plt.figure('uv')
#    plt.xlim(-1.1,1.1)
#    plt.ylim(-1.1,1.1)
#    for j in xrange(Nt):
#        plt.plot(upq[:,j], vpq[:,j], 'xr')

    # do DFT to get model visibilities
    Xpq = np.zeros([Na, Na, Nt], dtype=np.complex)
    Vpq = np.zeros([Na, Na, Nt], dtype=np.complex)
    for i, pq in enumerate(iter(pqlist)):
        p = int(pq[0])-1
        q = int(pq[1])-1
        gp = g[p,:]
        gqH = g[q,:].conj()
        for j in xrange(Nt):
            uv = np.array([upq[i,j], vpq[i,j]])
            K = np.exp(-2.0j*np.pi*np.dot(uv,lm))
            Xpq[p,q,j] = np.dot(K, IMflat)
            Xpq[q,p,j] = Xpq[p,q,j].conj()
            # corrupt model vis
            Vpq[p,q,j] = gp[j]*Xpq[p,q,j]*gqH[j] + sigman*np.random.randn() + sigman*1.0j*np.random.randn()
            Vpq[q,p,j] = Vpq[p,q,j].conj()

    # set operators and data structures for doing per antennae solve
    A = np.zeros([Na, Nt*Na, Nt], dtype=np.complex128) # to hold per-antenna response
    #Aobs = np.zeros([Na, Nt*Na, Nt], dtype=np.complex128) # to hold per-antenna obs response
    V = np.zeros([Na, Nt*Na], dtype=np.complex128) # to hold per-antenna data
    W = np.ones([Na, Nt*Na], dtype=np.float64) # to hold weights
    Sigma = np.zeros([Na, Na*Nt], dtype=np.complex128) # to hold per-antenna weights
    Sigmay = np.zeros([Na, Nt], dtype=np.complex128) # to hold diagonal of Ad.Sigmainv.A
    theta = np.zeros([Na, Nhypers]) # hyper-parameters excluding sigman
    Klist = []  # list holding covariance operators
    Kylist = [] # list holding Ky operators
    Dlist = []  # list holding D operators

    # construct response operator and solve iteratively in parallel for gbar
    deltheta = 1.0e-3
    diff = 1.0      # monitor update differences
    tol = 3.0e-3    # tolerance (max difference between updates)
    gbar = np.ones([Na, Nt], dtype=np.complex) # initial guess
    gobs = np.ones([Na, Nt], dtype=np.complex)  # initial guess
    i = 0 # iteration counter
    maxiter = 25 # maximum number of iterations
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
            if i==0:
                Sigma[p] = sigman**2 # should combine with weights for realistic data
                Sigmay[p] = np.diag(np.dot(A[p].T.conj(), np.diag(1.0/Sigma[p]).dot(A[p])))
                theta[p] = np.array([np.sqrt(2.0)*sigmaf, lGP]) #+ deltheta*np.random.randn(Nhypers))  # want common sigman
                Klist.append(ops.K_operator(t, np.append(theta[p,:], sigman), solve_mode="full", M=24, L=2.0, jit=1e-4))
                Kylist.append(ops.Ky_operator(Klist[p], Sigmay[p], solve_mode="full"))
                Dlist.append(ops.D_operator(Klist[p], Kylist[p]))
            else:
                Sigmay[p] = np.diag(np.dot(A[p].T.conj(), np.diag(1.0/Sigma[p]).dot(A[p])))
                Kylist[p].update(Klist[p], Sigmay[p])
                Dlist[p].update(Klist[p], Kylist[p])

        # Solve for mean
        gbar, gobs = algos.get_update(gold.copy(), gobsold.copy(), A, V, Sigma, Klist, Kylist, Dlist, i, Na)
        diff = np.max(np.abs(gbar-gold))

        i += 1
        print diff

    # test interpolation
    gp = algos.get_interp(theta, tp, gbar, gobs, Klist, Kylist, Dlist, Na)

    # plot result
    # plt.figure('phase GP')
    # gdiff = np.angle(gbar[0,:]) - np.angle(g[0,:])
    # plt.plot(t, np.angle(g[1,:]), 'b', label='True')
    # plt.plot(t, np.angle(gbar[1,:]) - gdiff, 'g', label='Learnt')
    # plt.plot(t, np.angle(gobs[1, :]) - gdiff, 'g', label='Observed')
    # #plt.plot(tp, np.angle(gp[1, :]) - gdiff, 'g', label='Observed')
    # plt.legend()
    # plt.savefig('/home/landman/Projects/SmoothCal/figures/phase.png', dpi=250)
    #
    # plt.figure('amp GP')
    # plt.plot(t, np.abs(g[1,:]), 'b', label='True')
    # plt.plot(t, np.abs(gbar[1,:]), 'g', label='Learnt')
    # plt.plot(t, np.abs(gobs[1, :]), 'r', label='Observed')
    # plt.plot(tp, np.abs(gp[1, :]), 'k', label='Interpolated')
    # plt.legend()
    # plt.savefig('/home/landman/Projects/SmoothCal/figures/amp.png', dpi = 250)


    plt.figure('g.real')
    plt.plot(tfull, (gfull[0,:]*gfull[1,:].conj()).real, 'k', label='True')
    plt.plot(t, (gbar[0,:]*gbar[1,:].conj()).real, 'b_', alpha=0.5, label='Learnt')
    #plt.plot(t, (gobs[0,:]*gobs[1, :].conj()).real, 'ro', label='Observed')
    plt.plot(tp, (gp[0,:]*gp[1, :].conj()).real, 'rx', alpha=0.5, label='Interpolated')
    plt.legend()
    plt.savefig('/home/landman/Projects/SmoothCal/figures/real.png', dpi = 250)

    plt.figure('g.imag')
    plt.plot(tfull, (gfull[0,:]*gfull[1,:].conj()).imag, 'k', label='True')
    plt.plot(t, (gbar[0,:]*gbar[1,:].conj()).imag, 'b_', alpha=0.5, label='Learnt')
    #plt.plot(t, (gobs[0,:]*gobs[1, :].conj()).imag, 'r', label='Observed')
    plt.plot(tp, (gp[0,:]*gp[1, :].conj()).imag, 'rx', alpha=0.5, label='Interpolated')
    plt.legend()
    plt.savefig('/home/landman/Projects/SmoothCal/figures/imag.png', dpi = 250)

    plt.show()

    # # solve for hypers
    # # set bounds for hypers
    # bnds = ((1e-5, None), (1e-5, None))
    # for i in xrange(Na-1):
    #     bnds += ((1e-5, None), (1e-5, None))
    # # set bound on sigman
    # bnds += ((1e-4, None),)
    # # set prior mean
    # gmean = np.ones([Na, Nt], dtype=np.complex)
    # # set starting guess for theta
    # # theta0 = np.append(theta.flatten(), np.array([sigman]))
    # # thetap = opt.fmin_l_bfgs_b(get_hypers, theta0, args=(V, A, W, Klist, Kylist, Dlist, gmean), approx_grad=1, bounds=bnds)
    # #
    # # print "Starting guess = ", theta0
    # # print "Estimated theta = ", thetap[0]
    # # lGP = 0.5
    # # sigmaf = 0.25
    # # sigman = 0.1
    # thetas2 = np.array([1.4, lGP, sigman])
    # thetas3 = np.array([sigmaf, 0.65, sigman])
    # thetas4 = np.array([sigmaf, lGP, 0.15])
    # thetas5 = np.array([0.5, 0.5, 0.099])
    # thetas = np.array([sigmaf, lGP, sigman])
    # i = 0
    # bnds = ((0.1, 0.5), (1e-5, None), (0.24, 0.26))
    # j = np.dot(A[i].T.conj(), V[i]*W[i])
    # Sigmay = np.diag(np.dot(A[i].T.conj(), np.diag(W[i]).dot(A[i])))
    # print "1 - "
    # H = train_impl(thetas, j, Sigmay, W[i], V[i], Klist[i], Kylist[i], Dlist[i], gmean[i])
    # print "2 - "
    # H2 = train_impl(thetas4, j, Sigmay, W[i], V[i], Klist[i], Kylist[i], Dlist[i], gmean[i])
    # print H < H2
    # thetap = opt.fmin_l_bfgs_b(train_impl, thetas5, args=(j, Sigmay, W[i], V[i], Klist[i], Kylist[i], Dlist[i], gmean[i]), fprime=None, bounds=bnds, m=25, factr=10.0, pgtol=1e-6, maxls=50)
    #                            #fprime=None, bounds=bnds) #, m=25, factr=1e4, pgtol=1e-6, maxls=50 approx_grad=True, epsilon=1.0e-2
    #
    # print thetap
    # thetas = np.array([np.sqrt(2.0)*sigmaf, lGP, sigman])
    # # train_impl(thetas, j, Sigmay, W[i], V[i], Klist[i], Kylist[i], Dlist[i], gmean[i])
    #
    # # thetastar = thetap[0]
    #
    # Nsamps = 50000
    # gsamps = draw_samples.draw_samples(meanfr, t, theta0, cov_func, Nsamps) + 1.0j * draw_samples.draw_samples(meanfi, t, theta0,
    #                                                                                                   cov_func, Nsamps)
    #
    # # get empirical distribution
    # gsamps -= np.ones(Nt, dtype=np.complex)
    # gcov = np.dot(gsamps.conj().T, gsamps)/(Nsamps - 1)
    #
    # plt.figure('test cov')
    # plt.plot(t, gcov[:, Nt//2])
    # plt.plot(t, cov_func(t, thetas))
    # plt.show()





