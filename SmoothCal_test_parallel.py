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
from pyrap.tables import table as pt


if __name__=="__main__":
    # open MS
    ms = pt("/home/landman/Projects/Data/MS_dir/PKS1934/channel0.ms", readonly=False)

    # get time stamps
    times = ms.getcol("TIME")
    t = np.unique(times)
    Nt = t.size

    # scale and shift times
    tnorm = (t - np.min(t))/np.max(t) - 0.5 # now tnorm lies between -0.5 and 0.5

    # get antennae list
    anta = ms.getcol("ANTENNA1")
    antb = ms.getcol("ANTENNA2")
    # number of antennae
    Na = antb[-1]+1

    # get single channel/correlation data
    ch = 0
    corr = 0
    mod_vis = ms.getcol("MODEL_DATA")[:, ch, corr]
    vis = ms.getcol("DATA")[:, ch, corr]

    # get flags
    flags = ms.getcol("FLAG")[:, ch, corr]

    # get weights
    weights = ms.getcol("WEIGHT")[:,0]

    # amplitude of calibrator taken from PKS document
    Flux = 15.0578061029
    mod_vis *= Flux

    # set initial guess for covariance params
    lGP = 0.5
    sigmaf = 0.15
    sigman = 0.25
    theta0 = np.array([sigmaf, lGP, sigman])
    Nhypers = theta0.size

    # build cubes
    Xpq = np.zeros([Na, Na, Nt], dtype=np.complex128) # model vis cube
    Vpq = np.zeros([Na, Na, Nt], dtype=np.complex128) # vis cube
    Wpq = np.ones_like(Vpq, dtype=np.float64) # weights cube
    for i in xrange(Nt):
        # get indices corresponding to time i
        I = np.argwhere(times == t[i]).squeeze()
        # antennae labels for time i
        ant1 = anta[I].squeeze()
        ant2 = antb[I].squeeze()
        # get visibilities
        mvis = mod_vis[I]
        dvis = vis[I]
        W = weights[I]
        F = flags[I]
        for j in xrange(I.size):
            f = F[j]
            if not f:
                p = ant1[j]
                q = ant2[j]
                Xpq[p, q, i] = mvis[j]
                Vpq[p, q, i] = dvis[j]
                Wpq[p, q, i] = W[j]
                Wpq[q, p, i] = W[j]

    # create masked arrays
    Xpq = ma.masked_array(Xpq, 0)
    Vpq = ma.masked_array(Vpq, 0)
    Wpq = ma.masked_array(Wpq, 0)

    # run Smoothcal cycle
    theta0[0] = np.sqrt(2)*sigmaf
    #theta0[-1] = np.sqrt(2) * sigman
    gbar, gobs, Klist, Kylist, Dlist, theta = algos.SmoothCal(Na, Nt, Xpq, Vpq, Wpq, tnorm, theta0, tol=5e-3, maxiter=25)

    #
    # # Do interpolation
    # meanval = np.mean(gbar, axis=1)
    # gp, gcov = algos.get_interp(theta, tfull, meanval, gobs, Klist, Kylist, Dlist, Na)
    #
    # # do StefCal cycle
    # gbar2, Sigmay = algos.StefCal(Na, Nt, Xpq, Vpq, Wpq, t, tol=5e-3, maxiter=25)
    #
    # # interpolate using StefCal data
    # for i in xrange(Na):
    #     Kylist[i].update(Klist[i], Sigmay[i])
    # meanval2 = np.mean(gbar2, axis=1)
    # gp2, gcov2 = algos.get_interp(theta, tfull, meanval, gbar2, Klist, Kylist, Dlist, Na)
    #
    # # do linear interpolation on StefCal result
    # gp3 = np.zeros_like(gp, dtype=np.complex128)
    # for i in xrange(Na):
    #     gp3[i,:] = np.interp(tfull, t, gbar2[i,:].real) + 1.0j*np.interp(tfull, t, gbar2[i,:].imag)
    #
    #
    # plot gains
    plt.figure('g.real')
    #plt.plot(tfull, (gfull[0,:]*gfull[1,:].conj()).real, 'k', label='True')
    #plt.plot(tfull[I], (gp[0,I]*gp[1,I].conj()).real, 'b+', alpha=0.5, label='SmoothCal')
    plt.plot(t, (gbar[0,:]*gbar[1, :].conj()).real, 'g--', alpha=0.5, label='StefCal')
    #plt.plot(tfull[I2], (gp[0,I2]*gp[1, I2].conj()).real, 'r+', alpha=0.5, label='Interpolated')
    plt.xlabel(r'$t$', fontsize=18)
    plt.ylabel(r'$Real(g_p g_q^\dagger)$', fontsize=18)
    #plt.plot(t, (gobs[0, :] * gobs[1, :].conj()).real, 'g--', alpha=0.5, label='Observed')
    plt.legend()
    plt.savefig('/home/landman/Projects/SmoothCal/figures/real.png', dpi = 250)

    plt.figure('g.imag')
    #plt.plot(tfull, (gfull[0,:]*gfull[1,:].conj()).imag, 'k', label='True')
    #plt.plot(tfull[I], (gp[0,I]*gp[1,I].conj()).imag, 'b+', alpha=0.5, label='SmoothCal')
    plt.plot(t, (gbar[0, :] * gbar[1, :].conj()).imag, 'g--', alpha=0.5, label='StefCal')
    #plt.plot(tfull[I2], (gp[0,I2]*gp[1, I2].conj()).imag, 'r+', alpha=0.5, label='Interpolated')
    plt.xlabel(r'$t$', fontsize=18)
    plt.ylabel(r'$Imag(g_p g_q^\dagger)$', fontsize=18)
    #plt.plot(t, (gobs[0, :] * gobs[1, :].conj()).imag, 'g--', alpha=0.5, label='Observed')
    plt.legend()
    plt.savefig('/home/landman/Projects/SmoothCal/figures/imag.png', dpi = 250)

    #
    #
    #
    # # plot errors
    # plt.figure('error')
    # plt.plot(tfull[I], np.abs(gfull[0, I] * gfull[1, I].conj() - gbar[0, :] * gbar[1, :].conj()), 'k.', label='SmoothCal')
    # plt.plot(tfull, np.abs(gfull[0, :] * gfull[1, :].conj() - gp2[0, :] * gp2[1, :].conj()), 'g--', label='Smoothed StefCal')
    # plt.plot(tfull, np.abs(gfull[0, :] * gfull[1, :].conj() - gp3[0, :] * gp3[1, :].conj()), 'b--', label='StefCal')
    # plt.plot(tfull, np.abs(gfull[0, :] * gfull[1, :].conj() - gp[0, :] * gp[1, :].conj()), 'k--', label='SmoothCal interp')
    # plt.fill_between(tfull, np.sqrt(np.diag(gcov2[0]).real + np.diag(gcov2[1]).real), np.zeros(Nfull), facecolor='b', alpha=0.5)
    # plt.xlabel(r'$t$', fontsize=18)
    # plt.ylabel(r'$|\epsilon|$', fontsize=18)
    # plt.legend()
    # plt.savefig('/home/landman/Projects/SmoothCal/figures/error.png', dpi = 250)
    #
    # plt.show()
    #
    #
    #
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





