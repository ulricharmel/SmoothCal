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


if __name__=="__main__":
    Nfull = 550
    tfull = np.linspace(-5.5, 5.5, Nfull)
    # set time domain
    interval = 50
    Nt = 300
    I = np.arange(interval)  # data index
    I2 = np.arange(interval, 2*interval)
    t = tfull[0:interval]
    tp = tfull[interval:2*interval]
    for i in xrange(2,Nfull//interval):
        if i%2==0:
            I = np.append(I, np.arange(i*interval,(i+1)*interval))
            t = np.append(t, tfull[i*interval:(i+1)*interval])
        else:
            I2 = np.append(I2, np.arange(i * interval, (i + 1) * interval))
            tp = np.append(tp, tfull[i * interval:(i + 1) * interval])

    # number of antennae
    Na = 3

    # set mean functions for amplitude and phase
    meanfr = np.ones(Nfull, dtype=np.float64)
    meanfi = np.zeros(Nfull, dtype=np.float64)

    # set covariance params
    lGP = 0.5
    sigmaf = 0.5
    sigman = 0.5
    theta0 = np.array([sigmaf, lGP, sigman])
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

    Wpq = np.ones_like(Vpq, dtype=np.float64)

    # run Smoothcal cycle
    gbar, gobs, Klist, Kylist, Dlist, theta = algos.SmoothCal(Na, Nt, Xpq, Vpq, Wpq, t, theta0, tol=1e-4, maxiter=25)

    # Do interpolation
    meanval = np.mean(gbar, axis=1)
    gp = algos.get_interp(theta, tfull, meanval, gobs, Klist, Kylist, Dlist, Na)

    # do StefCal cycle
    gbar2, Sigmay = algos.StefCal(Na, Nt, Xpq, Vpq, Wpq, t, tol=1e-4, maxiter=25)

    # interpolate using StefCal data
    for i in xrange(Na):
        Kylist[i].update(Klist[i], Sigmay[i])
    meanval2 = np.mean(gbar2, axis=1)
    gp2 = algos.get_interp(theta, tfull, meanval, gbar2, Klist, Kylist, Dlist, Na)

    # do linear interpolation on StefCal result
    gp3 = np.zeros_like(gp, dtype=np.complex128)
    for i in xrange(Na):
        gp3[i,:] = np.interp(tfull, t, gbar2[i,:].real) + 1.0j*np.interp(tfull, t, gbar2[i,:].imag)


    # plot gains
    plt.figure('g.real')
    plt.plot(tfull, (gfull[0,:]*gfull[1,:].conj()).real, 'k', label='True')
    plt.plot(tfull[I], (gp[0,I]*gp[1,I].conj()).real, 'b+', alpha=0.5, label='SmoothCal')
    plt.plot(t, (gbar2[0,:]*gbar2[1, :].conj()).real, 'g--', alpha=0.5, label='StefCal')
    plt.plot(tfull[I2], (gp[0,I2]*gp[1, I2].conj()).real, 'r+', alpha=0.5, label='Interpolated')
    #plt.plot(t, (gobs[0, :] * gobs[1, :].conj()).real, 'g--', alpha=0.5, label='Observed')
    plt.legend()
    plt.savefig('/home/landman/Projects/SmoothCal/figures/real.png', dpi = 250)

    plt.figure('g.imag')
    plt.plot(tfull, (gfull[0,:]*gfull[1,:].conj()).imag, 'k', label='True')
    plt.plot(tfull[I], (gp[0,I]*gp[1,I].conj()).imag, 'b+', alpha=0.5, label='SmoothCal')
    plt.plot(t, (gbar2[0, :] * gbar2[1, :].conj()).imag, 'g--', alpha=0.5, label='StefCal')
    plt.plot(tfull[I2], (gp[0,I2]*gp[1, I2].conj()).imag, 'r+', alpha=0.5, label='Interpolated')
    #plt.plot(t, (gobs[0, :] * gobs[1, :].conj()).imag, 'g--', alpha=0.5, label='Observed')
    plt.legend()
    plt.savefig('/home/landman/Projects/SmoothCal/figures/imag.png', dpi = 250)



    # plot errors
    plt.figure('error')
    plt.plot(tfull[I], np.abs(gfull[0, I] * gfull[1, I].conj() - gbar[0, :] * gbar[1, :].conj()), 'k.', label='SmoothCal')
    plt.plot(tfull, np.abs(gfull[0, :] * gfull[1, :].conj() - gp2[0, :] * gp2[1, :].conj()), 'g--', label='Smoothed StefCal')
    plt.plot(tfull, np.abs(gfull[0, :] * gfull[1, :].conj() - gp3[0, :] * gp3[1, :].conj()), 'b--', label='StefCal')
    plt.plot(tfull, np.abs(gfull[0, :] * gfull[1, :].conj() - gp[0, :] * gp[1, :].conj()), 'k--', label='SmoothCal interp')
    plt.legend()
    plt.savefig('/home/landman/Projects/SmoothCal/figures/error.png', dpi = 250)

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





