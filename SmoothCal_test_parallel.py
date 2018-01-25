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

def R(IM, upq, vpq, lm, pqlist, Xpq):
    """
    The response operator coded as a DFT
    :param IM: Npix x Npix array containing model image
    :param upq: Na x Nt array of baseline coordinates
    :param vpq: Na x Nt array of baseline coordinates
    :param lm: 2 x Npix**2 array of sky coordinates
    :param pqlist: a list of antennae pairs (used for the iterator)
    :param Xpq: Na x Na x Nt array to hold model visibilities
    :return: Xpq the model visibilities
    """
    IMflat = IM.flatten()
    for i, pq in enumerate(iter(pqlist)):
        p = int(pq[0])-1
        q = int(pq[1])-1
        gp = g[p,:]
        gqH = g[q,:].conj()
        for j in xrange(Nt):
            uv = np.array([upq[i,j], vpq[i,j]])
            K = np.exp(-2.0j*np.pi*np.dot(uv,lm))
            #print K.shape, uv.shape, lm.shape, IMflat.shape
            Xpq[p,q,j] = np.dot(K, IMflat)
            Xpq[q,p,j] = Xpq[p,q,j].conj()
    return Xpq

def RH(Xpq, upq, vpq, lm, ID):
    """
    The adjoint of the DFT response operator
    :param Xpq: Na x Na x Nt array containing model visibilities
    :param upq: Na x Nt array of baseline coordinates
    :param vpq: Na x Nt array of baseline coordinates
    :param lm: 2 x Npix**2 array of sky coordinates
    :param ID: Npix x Npix array to hold resulting image
    :return: 
    """
    ID_flat = ID.flatten()
    for i, pq in enumerate(iter(pqlist)):
        p = int(pq[0]) - 1
        q = int(pq[1]) - 1
        gp = g_target[p, :]
        gqH = g_target[q, :].conj()
        uv = np.vstack((upq_target[i, :], vpq_target[i, :]))
        X = Vpq_target[p, q, :]
        K = np.exp(-2.0j * np.pi * np.dot(lm.T, uv.conj()))
        ID_flat += np.dot(K, X).real
    ID = ID_flat.reshape(Npix, Npix)
    return ID

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
            I = np.append(I, np.arange(i*interval,(i+1)*interval)) #times indices of calibrator field
            t = np.append(t, tfull[i*interval:(i+1)*interval])
        else:
            I2 = np.append(I2, np.arange(i * interval, (i + 1) * interval)) #time indices of target field
            tp = np.append(tp, tfull[i * interval:(i + 1) * interval])

    # number of antennae
    Na = 4

    # set mean functions for real and imaginary parts
    meanfr = np.ones(Nfull, dtype=np.float64)
    meanfi = np.zeros(Nfull, dtype=np.float64)

    # set covariance params
    lGP = 0.5
    sigmaf = 0.15
    sigman = 0.25
    theta0 = np.array([sigmaf, lGP, sigman])
    Nhypers = theta0.size


    # set covariance function and sample gains
    # Mattern 5/2
    # def cov_func(x, theta):
    #         return theta[0] ** 2 * np.exp(-np.sqrt(5) * np.abs(x) / theta[1]) * (1 + np.sqrt(5) * np.abs(x) / theta[1] + 5 * np.abs(x) ** 2 / (3 * theta[1] ** 2))

    # exponential squared
    def cov_func(x, theta):
        return theta[0]**2*np.exp(-x**2/(2*theta[1]**2))
    # draw some random gain realisations
    gfull = draw_samples.draw_samples(meanfr, tfull, theta0, cov_func, Na) + 1.0j*draw_samples.draw_samples(meanfi, tfull, theta0, cov_func, Na)

    # get true gains for calibrator field
    g = np.zeros([Na, Nt], dtype=np.complex128)
    # for i in xrange(0,Nfull//interval - Nfull//(2*interval)):
    #     g[:, i*interval:(i+1)*interval] = gfull[:, 2*i*interval:(2*i+1)*interval]
    g = gfull[:, I]

    # get true gains for target field
    N_target = Nfull - Nt
    g_target = np.zeros([Na, N_target], dtype=np.complex128)
    g_target = gfull[:, I2]

    # make sky model for calibration (calibrator field)
    Npix = 65
    lmax = 1.0
    mmax = 1.0
    l = np.linspace(-lmax, lmax, Npix)
    m = np.linspace(-mmax, mmax, Npix)
    ll, mm = np.meshgrid(l, m)
    lm = (np.vstack((ll.flatten(), mm.flatten())))
    IM = np.zeros([Npix, Npix])
    IM[Npix//2, Npix//2] = 10.0
    IM[Npix//4, Npix//4] = 1.0
    IM[3*Npix//4, 3*Npix//4] = 1.0
    IM[Npix//4, 3*Npix//4] = 1.0
    IM[3*Npix//4, Npix//4] = 1.0
    IMflat = IM.flatten()

    # make sky model for imaging (target field)
    IM2 = np.zeros([Npix, Npix])
    IM2[Npix//2, Npix//2] = 1.0
    IM2[Npix//4, Npix//4] = 0.5
    IM2[3*Npix//4, 3*Npix//4] = 0.25
    IM2[Npix//4, 3*Npix//4] = 0.1
    IM2[3*Npix//4, Npix//4] = 0.05
    IM2flat = IM2.flatten()

    # this is to create the pq iterator (only works for N<10 antennae)
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
    u = 10*np.random.random(Na)
    v = 10*np.random.random(Na)

    # create calibration baselines with time axis
    upq = np.zeros([N, Nt])
    vpq = np.zeros([N, Nt])
    phi_full = np.linspace(0, np.pi, Nfull) # to simulate earth rotation
    phi = phi_full[I]
    for i, pq in enumerate(iter(pqlist)):
        print i, pq
        upq[i, 0] = u[int(pq[0])-1] - u[int(pq[1])-1]
        vpq[i, 0] = v[int(pq[0])-1] - v[int(pq[1])-1]
        for j in xrange(1, Nt):
            rottheta = np.array([[np.cos(phi[j]), -np.sin(phi[j])], [np.sin(phi[j]), np.cos(phi[j])]])
            upq[i, j], vpq[i, j] = np.dot(rottheta, np.array([upq[i, 0], vpq[i, 0]]))

    # create target baselines with time axis
    upq_target = np.zeros([N, N_target])
    vpq_target = np.zeros([N, N_target])
    phi_target = phi_full[I2]
    for i, pq in enumerate(iter(pqlist)):
        print i, pq
        upq_ref = u[int(pq[0]) - 1] - u[int(pq[1]) - 1]
        vpq_ref = v[int(pq[0]) - 1] - v[int(pq[1]) - 1]
        for j in xrange(0, N_target):
            rottheta = np.array([[np.cos(phi_target[j]), -np.sin(phi_target[j])], [np.sin(phi_target[j]), np.cos(phi_target[j])]])
            upq_target[i, j], vpq_target[i, j] = np.dot(rottheta, np.array([upq_ref, vpq_ref]))

    # inspect uv-coverage
    plt.figure('uv')
    plt.xlim(-10.1,10.1)
    plt.ylim(-10.1,10.1)
    # calibrator baselines
    for j in xrange(Nt):
        plt.plot(upq[:, j], vpq[:, j], 'xr')
    # target baselines
    for j in xrange(N_target):
        plt.plot(upq_target[:, j], vpq_target[:, j], 'xb')

    #plt.show()

    # do DFT to get calibration model visibilities
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
            #print K.shape, uv.shape, lm.shape, IMflat.shape
            Xpq[p,q,j] = np.dot(K, IMflat)
            Xpq[q,p,j] = Xpq[p,q,j].conj()
            # corrupt model vis
            Vpq[p,q,j] = gp[j]*Xpq[p,q,j]*gqH[j] + sigman*np.random.randn() + sigman*1.0j*np.random.randn()
            Vpq[q,p,j] = Vpq[p,q,j].conj()

    #sys.exit()
    # do DFT to get target model visibilities
    Xpq_target = np.zeros([Na, Na, N_target], dtype=np.complex)
    Vpq_target = np.zeros([Na, Na, N_target], dtype=np.complex)
    for i, pq in enumerate(iter(pqlist)):
        p = int(pq[0]) - 1
        q = int(pq[1]) - 1
        gp = g_target[p, :]
        gqH = g_target[q, :].conj()
        for j in xrange(N_target):
            uv = np.array([upq_target[i, j], vpq_target[i, j]])
            K = np.exp(-2.0j * np.pi * np.dot(uv, lm))
            Xpq_target[p, q, j] = np.dot(K, IM2flat)
            Xpq_target[q, p, j] = Xpq_target[p, q, j].conj()
            # corrupt model vis
            Vpq_target[p, q, j] = gp[j] * Xpq_target[p, q, j] * gqH[
                j] + sigman * np.random.randn() + sigman * 1.0j * np.random.randn()
            Vpq_target[q, p, j] = Vpq_target[p, q, j].conj()

    # set weights
    Wpq = np.ones_like(Vpq, dtype=np.float64)

    # run Smoothcal cycle
    theta0[0] = np.sqrt(2)*sigmaf
    #theta0[-1] = np.sqrt(2) * sigman
    gbar, gobs, Klist, Kylist, Dlist, theta = algos.SmoothCal(Na, Nt, Xpq, Vpq, Wpq, t, theta0, tol=5e-3, maxiter=25)

    # Do interpolation
    meanval = np.mean(gbar, axis=1)
    gp, gcov = algos.get_interp(theta, tfull, meanval, gobs, Klist, Kylist, Dlist, Na)

    # do StefCal cycle
    gbar2, Sigmay = algos.StefCal(Na, Nt, Xpq, Vpq, Wpq, t, tol=5e-3, maxiter=25)

    # interpolate using StefCal data
    for i in xrange(Na):
        Kylist[i].update(Klist[i], Sigmay[i])
    meanval2 = np.mean(gbar2, axis=1)
    gp2, gcov2 = algos.get_interp(theta, tfull, meanval, gbar2, Klist, Kylist, Dlist, Na)

    # do linear interpolation on StefCal result
    gp3 = np.zeros_like(gp, dtype=np.complex128)
    for i in xrange(Na):
        gp3[i,:] = np.interp(tfull, t, gbar2[i,:].real) + 1.0j*np.interp(tfull, t, gbar2[i,:].imag)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 9))
    ax[0].plot(tfull, (gfull[0,:]*gfull[1,:].conj()).real, 'k', label='True')
    ax[0].plot(tfull[I], (gp[0,I]*gp[1,I].conj()).real, 'b+', alpha=0.5, label='SmoothCal')
    ax[0].plot(t, (gbar2[0,:]*gbar2[1, :].conj()).real, 'g--', alpha=0.5, label='StefCal')
    ax[0].plot(tfull[I2], (gp[0,I2]*gp[1, I2].conj()).real, 'r+', alpha=0.5, label='Interpolated')
    ax[0].set_xlabel(r'$t$', fontsize=18)
    ax[0].set_ylabel(r'$Real(g_p g_q^\dagger)$', fontsize=18)
    #ax[0].legend()

    ax[1].plot(tfull, (gfull[0, :] * gfull[1, :].conj()).imag, 'k', label='True')
    ax[1].plot(tfull[I], (gp[0, I] * gp[1, I].conj()).imag, 'b+', alpha=0.5, label='SmoothCal')
    ax[1].plot(t, (gbar2[0, :] * gbar2[1, :].conj()).imag, 'g--', alpha=0.5, label='StefCal')
    ax[1].plot(tfull[I2], (gp[0, I2] * gp[1, I2].conj()).imag, 'r+', alpha=0.5, label='Interpolated')
    ax[1].set_xlabel(r'$t$', fontsize=18)
    ax[1].set_ylabel(r'$Imag(g_p g_q^\dagger)$', fontsize=18)
    # plt.plot(t, (gobs[0, :] * gobs[1, :].conj()).imag, 'g--', alpha=0.5, label='Observed')
    ax[1].legend(loc=2)

    fig.savefig('/home/landman/Projects/SmoothCal/figures/Sim_combined.png', dpi = 250)

    # # plot gains
    # plt.figure('g.real')
    # plt.plot(tfull, (gfull[0,:]*gfull[1,:].conj()).real, 'k', label='True')
    # plt.plot(tfull[I], (gp[0,I]*gp[1,I].conj()).real, 'b+', alpha=0.5, label='SmoothCal')
    # plt.plot(t, (gbar2[0,:]*gbar2[1, :].conj()).real, 'g--', alpha=0.5, label='StefCal')
    # plt.plot(tfull[I2], (gp[0,I2]*gp[1, I2].conj()).real, 'r+', alpha=0.5, label='Interpolated')
    # plt.xlabel(r'$t$', fontsize=18)
    # plt.ylabel(r'$Real(g_p g_q^\dagger)$', fontsize=18)
    # #plt.plot(t, (gobs[0, :] * gobs[1, :].conj()).real, 'g--', alpha=0.5, label='Observed')
    # plt.legend()
    # plt.savefig('/home/landman/Projects/SmoothCal/figures/Sim_real.png', dpi = 250)
    #
    # plt.figure('g.imag')
    # plt.plot(tfull, (gfull[0,:]*gfull[1,:].conj()).imag, 'k', label='True')
    # plt.plot(tfull[I], (gp[0,I]*gp[1,I].conj()).imag, 'b+', alpha=0.5, label='SmoothCal')
    # plt.plot(t, (gbar2[0, :] * gbar2[1, :].conj()).imag, 'g--', alpha=0.5, label='StefCal')
    # plt.plot(tfull[I2], (gp[0,I2]*gp[1, I2].conj()).imag, 'r+', alpha=0.5, label='Interpolated')
    # plt.xlabel(r'$t$', fontsize=18)
    # plt.ylabel(r'$Imag(g_p g_q^\dagger)$', fontsize=18)
    # #plt.plot(t, (gobs[0, :] * gobs[1, :].conj()).imag, 'g--', alpha=0.5, label='Observed')
    # plt.legend()
    # plt.savefig('/home/landman/Projects/SmoothCal/figures/Sim_imag.png', dpi = 250)



    # plot errors
    plt.figure('error')
    plt.plot(tfull[I], np.abs(gfull[0, I] * gfull[1, I].conj() - gbar[0, :] * gbar[1, :].conj()), 'k.', label='SmoothCal')
    plt.plot(tfull, np.abs(gfull[0, :] * gfull[1, :].conj() - gp2[0, :] * gp2[1, :].conj()), 'g--', label='Smoothed StefCal')
    plt.plot(tfull, np.abs(gfull[0, :] * gfull[1, :].conj() - gp3[0, :] * gp3[1, :].conj()), 'b--', label='StefCal')
    plt.plot(tfull, np.abs(gfull[0, :] * gfull[1, :].conj() - gp[0, :] * gp[1, :].conj()), 'k--', label='SmoothCal interp')
    plt.fill_between(tfull, np.sqrt(np.diag(gcov[0]).real + np.diag(gcov[1]).real), np.zeros(Nfull), facecolor='b', alpha=0.5)
    plt.xlabel(r'$t$', fontsize=18)
    plt.ylabel(r'$|\epsilon|$', fontsize=18)
    plt.legend()
    plt.savefig('/home/landman/Projects/SmoothCal/figures/Sim_error.png', dpi = 250)

    #plt.show()



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


    # make the PSF
    Wpq_target = np.ones_like(Vpq_target, dtype=np.float64) # need to propagate uncertainties on gains eventually
    lPSF = np.linspace(-2*lmax, 2*lmax, 2*Npix-1)
    mPSF = np.linspace(-2*mmax, 2*mmax, 2*Npix-1)
    llPSF, mmPSF = np.meshgrid(lPSF, mPSF)
    lmPSF = (np.vstack((llPSF.flatten(), mmPSF.flatten())))
    PSF = np.zeros([2*Npix-1, 2*Npix-1])
    PSFflat = PSF.flatten()
    print "Making PSF"
    for i, pq in enumerate(iter(pqlist)):
        p = int(pq[0]) - 1
        q = int(pq[1]) - 1
        gp = g_target[p, :]
        gqH = g_target[q, :].conj()
        uv = np.vstack((upq_target[i, :], vpq_target[i, :]))
        W = Wpq_target[p, q, :]
        K = np.exp(-2.0j * np.pi * np.dot(lmPSF.T, uv.conj()))
        PSFflat += np.dot(K, W).real

    PSFmax = PSFflat.max()
    PSFflat /= PSFmax
    PSF = PSFflat.reshape(2*Npix-1, 2*Npix-1)
    plt.figure('PSF')
    plt.imshow(PSFflat.reshape(2*Npix-1, 2*Npix-1))
    plt.colorbar()

    # make the dirty image
    ID = np.zeros([Npix, Npix])
    ID_flat = ID.flatten()
    print "Making Dirty"
    for i, pq in enumerate(iter(pqlist)):
        p = int(pq[0]) - 1
        q = int(pq[1]) - 1
        gp = g_target[p, :]
        gqH = g_target[q, :].conj()
        uv = np.vstack((upq_target[i, :], vpq_target[i, :]))
        X = Vpq_target[p, q, :]
        K = np.exp(-2.0j * np.pi * np.dot(lm.T, uv.conj()))
        ID_flat += np.dot(K, X).real
    ID_flat /= PSFmax
    ID = ID_flat.reshape(Npix, Npix)

    plt.figure('Dirty')
    plt.imshow(ID)
    plt.colorbar()

    print "Cleaning"
    IM, IR = algos.Hogbom(ID, PSF)

    plt.figure("IM")
    plt.imshow(IM)
    plt.colorbar()

    plt.figure("IR")
    plt.imshow(IR)
    plt.colorbar()

    plt.show()





