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
from astropy.io import fits
from astropy import wcs

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
    Nt = upq.shape[1]
    for i, pq in enumerate(iter(pqlist)):
        p = int(pq[0])-1
        q = int(pq[1])-1
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
        uv = np.vstack((upq[i, :], vpq[i, :]))
        X = Xpq[p, q, :]
        K = np.exp(-2.0j * np.pi * np.dot(lm.T, uv.conj()))
        ID_flat += np.dot(K, X).real
    ID = ID_flat.reshape(Npix, Npix)
    return ID

if __name__=="__main__":
    Nfull = 950
    tfull = np.linspace(-5.5, 5.5, Nfull)
    # set time domain
    interval = 50
    Nt = 500
    I = np.arange(interval)  # data index
    I2 = np.arange(interval, 2*interval)
    t = tfull[0:interval]
    tp = tfull[interval:2*interval]
    for i in xrange(2,Nfull//interval):
        if i%2==0:
            I = np.append(I, np.arange(i*interval,(i+1)*interval)) #time indices of calibrator field
            t = np.append(t, tfull[i*interval:(i+1)*interval])
        else:
            I2 = np.append(I2, np.arange(i * interval, (i + 1) * interval)) #time indices of target field
            tp = np.append(tp, tfull[i * interval:(i + 1) * interval])

    # number of antennae
    Na = 9

    # set mean functions for real and imaginary parts
    meanfr = np.ones(Nfull, dtype=np.float64)
    meanfi = np.zeros(Nfull, dtype=np.float64)

    # set covariance params
    lGP = 0.35
    sigmaf = 0.05
    sigman = 0.1
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
    gfull_true = draw_samples.draw_samples(meanfr, tfull, theta0, cov_func, Na) + 1.0j*draw_samples.draw_samples(meanfi, tfull, theta0, cov_func, Na)

    # get true gains for calibrator field
    g = np.zeros([Na, Nt], dtype=np.complex128)
    g = gfull_true[:, I]

    # get true gains for target field
    N_target = Nfull - Nt
    g_target = np.zeros([Na, N_target], dtype=np.complex128)
    g_target = gfull_true[:, I2]

    # make sky model for calibration (calibrator field)
    Npix = 65
    lmax = 1.0
    mmax = 1.0
    l = np.linspace(-lmax, lmax, Npix)
    m = np.linspace(-mmax, mmax, Npix)
    ll, mm = np.meshgrid(l, m)
    lm = (np.vstack((ll.flatten(), mm.flatten())))
    IM = np.zeros([Npix, Npix])
    IM[Npix//2, Npix//2] = 100.0
    IM[Npix//4, Npix//4] = 10.0
    IM[3*Npix//4, 3*Npix//4] = 5.0
    IM[Npix//4, 3*Npix//4] = 2.5
    IM[3*Npix//4, Npix//4] = 1.0
    IMflat = IM.flatten()

    # make sky model for imaging (target field)
    IM_target = np.zeros([Npix, Npix])
    IM_target[Npix//2, Npix//2] = 50.0
    IM_target[Npix//4, Npix//4] = 5.0
    IM_target[3*Npix//4, 3*Npix//4] = 1.0
    IM_target[Npix//4, 3*Npix//4] = 0.5
    IM_target[3*Npix//4, Npix//4] = 0.1
    IM_target_flat = IM_target.flatten()

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
        #print i, pq
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
        #print i, pq
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

    plt.savefig('/home/landman/Projects/SmoothCal/figures/uv_coverage.png', dpi=250)

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
            Xpq_target[p, q, j] = np.dot(K, IM_target_flat)
            Xpq_target[q, p, j] = Xpq_target[p, q, j].conj()
            # corrupt model vis
            Vpq_target[p, q, j] = gp[j] * Xpq_target[p, q, j] * gqH[
                j] + sigman * np.random.randn() + sigman * 1.0j * np.random.randn()
            Vpq_target[q, p, j] = Vpq_target[p, q, j].conj()

    # set weights
    Wpq = np.ones_like(Vpq, dtype=np.float64)

    # run Smoothcal cycle
    theta0[0] = np.sqrt(2)*sigmaf
    theta0[-1] = np.sqrt(2) * sigman
    gbar_smooth, gobs_smooth, Klist, Kylist, Dlist, theta = algos.SmoothCal(Na, Nt, Xpq, Vpq, Wpq, t, theta0, tol=1.0e-4, maxiter=25)

    # Do interpolation
    meanval = np.mean(gbar_smooth, axis=1)
    gmean_smooth, gcov_smooth = algos.get_interp(theta, tfull, meanval, gobs_smooth, Klist, Kylist, Dlist, Na)

    # do StefCal cycle
    gbar_stef, Sigmay = algos.StefCal(Na, Nt, Xpq, Vpq, Wpq, t, tol=1.0e-4, maxiter=25)

    # interpolate using StefCal data
    for i in xrange(Na):
        Kylist[i].update(Klist[i], Sigmay[i])
    meanval2 = np.mean(gbar_stef, axis=1)
    gmean_stef, gcov_stef = algos.get_interp(theta, tfull, meanval, gbar_stef, Klist, Kylist, Dlist, Na)

    # do linear interpolation on StefCal result
    gbar_stef_lin_interp = np.zeros_like(gfull_true, dtype=np.complex128)
    for i in xrange(Na):
        gbar_stef_lin_interp[i,:] = np.interp(tfull, t, gbar_stef[i,:].real) + 1.0j*np.interp(tfull, t, gbar_stef[i,:].imag)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 9))
    ax[0].plot(tfull, (gfull_true[0]*gfull_true[1].conj()).real, 'k', label='True')
    ax[0].plot(t, (gbar_smooth[0]*gbar_smooth[1].conj()).real, 'b+', alpha=0.5, label='SmoothCal')
    ax[0].plot(t, (gbar_stef[0]*gbar_stef[1].conj()).real, 'g--', alpha=0.5, label='StefCal')
    ax[0].plot(tfull[I2], (gmean_smooth[0,I2]*gmean_smooth[1, I2].conj()).real, 'b+', alpha=0.5)
    # ax[0].plot(tfull[I2], (gmean_stef[0, I2] * gmean_stef[1, I2].conj()).real, 'g+', alpha=0.5,
    #            label='Stef Interpolated')
    ax[0].set_xlabel(r'$t$', fontsize=18)
    ax[0].set_ylabel(r'$Real(g_p g_q^\dagger)$', fontsize=18)
    #ax[0].legend()

    ax[1].plot(tfull, (gfull_true[0] * gfull_true[1].conj()).imag, 'k', label='True')
    ax[1].plot(t, (gbar_smooth[0] * gbar_smooth[1].conj()).imag, 'b+', alpha=0.5, label='SmoothCal')
    ax[1].plot(t, (gbar_stef[0] * gbar_stef[1].conj()).imag, 'g--', alpha=0.5, label='StefCal')
    ax[1].plot(tfull[I2], (gmean_smooth[0, I2] * gmean_smooth[1, I2].conj()).imag, 'b+', alpha=0.5)
    # ax[1].plot(tfull[I2], (gmean_stef[0, I2] * gmean_stef[1, I2].conj()).imag, 'g+', alpha=0.5,
    #            label='Stef Interpolated')
    ax[1].set_xlabel(r'$t$', fontsize=18)
    ax[1].set_ylabel(r'$Imag(g_p g_q^\dagger)$', fontsize=18)
    # plt.plot(t, (gobs[0, :] * gobs[1, :].conj()).imag, 'g--', alpha=0.5, label='Observed')
    ax[1].legend(loc=2)

    fig.savefig('/home/landman/Projects/SmoothCal/figures/Sim_combined.png', dpi = 250)

    # plot errors
    plt.figure('error')
    plt.plot(tfull[I], np.abs(gfull_true[0, I] * gfull_true[1, I].conj() - gbar_smooth[0, :] * gbar_smooth[1, :].conj()), 'k.', label='SmoothCal')
    plt.plot(tfull, np.abs(gfull_true[0, :] * gfull_true[1, :].conj() - gmean_stef[0, :] * gmean_stef[1, :].conj()), 'g--', label='Smoothed StefCal')
    plt.plot(tfull, np.abs(gfull_true[0, :] * gfull_true[1, :].conj() - gbar_stef_lin_interp[0, :] * gbar_stef_lin_interp[1, :].conj()), 'b--', label='StefCal')
    plt.plot(tfull, np.abs(gfull_true[0, :] * gfull_true[1, :].conj() - gmean_smooth[0, :] * gmean_smooth[1, :].conj()), 'k--', label='SmoothCal interp')
    plt.fill_between(tfull, np.sqrt(np.diag(gcov_smooth[0]).real + np.diag(gcov_smooth[1]).real), np.zeros(Nfull), facecolor='b', alpha=0.5)
    plt.xlabel(r'$t$', fontsize=18)
    plt.ylabel(r'$|\epsilon|$', fontsize=18)
    plt.legend()
    plt.savefig('/home/landman/Projects/SmoothCal/figures/Sim_error.png', dpi = 250)

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
    print "Making perfect PSF"
    for i, pq in enumerate(iter(pqlist)):
        p = int(pq[0]) - 1
        q = int(pq[1]) - 1
        uv = np.vstack((upq_target[i, :], vpq_target[i, :]))
        W = Wpq_target[p, q, :]
        K = np.exp(-2.0j * np.pi * np.dot(lmPSF.T, uv.conj()))
        PSFflat += np.dot(K, W).real

    PSFmax = PSFflat.max()
    PSFflat /= PSFmax
    PSF = PSFflat.reshape(2*Npix-1, 2*Npix-1)

    hdu = fits.PrimaryHDU(PSF)
    hdul = fits.HDUList([hdu])
    hdul.writeto('/home/landman/Projects/SmoothCal/figures/PSF.fits', overwrite=True)
    hdul.close()

    # make the dirty image
    ID = np.zeros([Npix, Npix])
    ID_flat = ID.flatten()
    print "Making perfect Dirty"
    for i, pq in enumerate(iter(pqlist)):
        p = int(pq[0]) - 1
        q = int(pq[1]) - 1
        uv = np.vstack((upq_target[i, :], vpq_target[i, :]))
        X = Xpq_target[p, q, :]
        K = np.exp(-2.0j * np.pi * np.dot(lm.T, uv.conj()))
        ID_flat += np.dot(K, X).real
    ID_flat /= PSFmax
    ID = ID_flat.reshape(Npix, Npix)

    print "Cleaning"
    IM, IR = algos.Hogbom(ID, PSF, peak_fact=1e-8)

    # save images to fits
    hdu = fits.PrimaryHDU(ID)
    hdul = fits.HDUList([hdu])
    hdul.writeto('/home/landman/Projects/SmoothCal/figures/ID.fits', overwrite=True)
    hdul.close()

    hdu = fits.PrimaryHDU(IM)
    hdul = fits.HDUList([hdu])
    hdul.writeto('/home/landman/Projects/SmoothCal/figures/IM.fits', overwrite=True)
    hdul.close()

    hdu = fits.PrimaryHDU(IR)
    hdul = fits.HDUList([hdu])
    hdul.writeto('/home/landman/Projects/SmoothCal/figures/IR.fits', overwrite=True)
    hdul.close()

    # Now try clean uncalibrated data
    # make the dirty image
    IDu = np.zeros([Npix, Npix])
    ID_flatu = ID.flatten()
    print "Making uncalibrated Dirty"
    for i, pq in enumerate(iter(pqlist)):
        p = int(pq[0]) - 1
        q = int(pq[1]) - 1
        uv = np.vstack((upq_target[i, :], vpq_target[i, :]))
        V = Vpq_target[p, q, :]
        K = np.exp(-2.0j * np.pi * np.dot(lm.T, uv.conj()))
        ID_flatu += np.dot(K, V).real
    ID_flatu /= PSFmax
    IDu = ID_flatu.reshape(Npix, Npix)

    print "Cleaning"
    IMu, IRu = algos.Hogbom(IDu, PSF, peak_fact=5e-3)

    # save images to fits
    hdu = fits.PrimaryHDU(IDu)
    hdul = fits.HDUList([hdu])
    hdul.writeto('/home/landman/Projects/SmoothCal/figures/IDu.fits', overwrite=True)
    hdul.close()

    hdu = fits.PrimaryHDU(IMu)
    hdul = fits.HDUList([hdu])
    hdul.writeto('/home/landman/Projects/SmoothCal/figures/IMu.fits', overwrite=True)
    hdul.close()

    hdu = fits.PrimaryHDU(IRu)
    hdul = fits.HDUList([hdu])
    hdul.writeto('/home/landman/Projects/SmoothCal/figures/IRu.fits', overwrite=True)
    hdul.close()


    ## Now apply StefCal gain solutions
    Xpq_corrected_Stef = np.zeros_like(Xpq_target)
    for i, pq in enumerate(iter(pqlist)):
        p = int(pq[0])-1
        q = int(pq[1])-1
        gptemp = gbar_stef_lin_interp[p, I2]
        gqtempH = gbar_stef_lin_interp[q, I2].conj()
        for j in xrange(N_target):
            Xpq_corrected_Stef[p, q, j] = Vpq_target[p, q, j]/(gptemp[j]*gqtempH[j])
            Xpq_corrected_Stef[q, p, j] = Xpq_corrected_Stef[p, q, j].conj()

    # image corrected vis
    ID_Stef = np.zeros([Npix, Npix])
    ID_flat_Stef = ID_Stef.flatten()
    print "Making StefCal Dirty"
    for i, pq in enumerate(iter(pqlist)):
        p = int(pq[0]) - 1
        q = int(pq[1]) - 1
        uv = np.vstack((upq_target[i, :], vpq_target[i, :]))
        V = Xpq_corrected_Stef[p, q, :]
        K = np.exp(-2.0j * np.pi * np.dot(lm.T, uv.conj()))
        ID_flat_Stef += np.dot(K, V).real
    ID_flat_Stef /= PSFmax
    ID_Stef = ID_flat_Stef.reshape(Npix, Npix)

    print "Cleaning"
    IM_Stef, IR_Stef = algos.Hogbom(ID_Stef, PSF, peak_fact=1.0e-3)

    # save images to fits
    hdu = fits.PrimaryHDU(ID_Stef)
    hdul = fits.HDUList([hdu])
    hdul.writeto('/home/landman/Projects/SmoothCal/figures/ID_Stef.fits', overwrite=True)
    hdul.close()

    hdu = fits.PrimaryHDU(IM_Stef)
    hdul = fits.HDUList([hdu])
    hdul.writeto('/home/landman/Projects/SmoothCal/figures/IM_Stef.fits', overwrite=True)
    hdul.close()

    hdu = fits.PrimaryHDU(IR_Stef)
    hdul = fits.HDUList([hdu])
    hdul.writeto('/home/landman/Projects/SmoothCal/figures/IR_Stef.fits', overwrite=True)
    hdul.close()

    ## SmoothCal gians without reweighting
    Xpq_corrected_Smooth = np.zeros_like(Xpq_target)
    for i, pq in enumerate(iter(pqlist)):
        p = int(pq[0])-1
        q = int(pq[1])-1
        gptemp = gmean_smooth[p, I2]
        gqtempH = gmean_smooth[q, I2].conj()
        for j in xrange(N_target):
            Xpq_corrected_Smooth[p, q, j] = Vpq_target[p, q, j]/(gptemp[j]*gqtempH[j])
            Xpq_corrected_Smooth[q, p, j] = Xpq_corrected_Stef[p, q, j].conj()

    # image corrected vis
    ID_Smooth = np.zeros([Npix, Npix])
    ID_flat_Smooth = ID_Smooth.flatten()
    print "Making SmoothCal Dirty"
    for i, pq in enumerate(iter(pqlist)):
        p = int(pq[0]) - 1
        q = int(pq[1]) - 1
        uv = np.vstack((upq_target[i, :], vpq_target[i, :]))
        V = Xpq_corrected_Smooth[p, q, :]
        K = np.exp(-2.0j * np.pi * np.dot(lm.T, uv.conj()))
        ID_flat_Smooth += np.dot(K, V).real
    ID_flat_Smooth /= PSFmax
    ID_Smooth = ID_flat_Smooth.reshape(Npix, Npix)

    print "Cleaning"
    IM_Smooth, IR_Smooth = algos.Hogbom(ID_Smooth, PSF, peak_fact=1.0e-3)

    # save images to fits
    hdu = fits.PrimaryHDU(ID_Smooth)
    hdul = fits.HDUList([hdu])
    hdul.writeto('/home/landman/Projects/SmoothCal/figures/ID_Smooth.fits', overwrite=True)
    hdul.close()

    hdu = fits.PrimaryHDU(IM_Smooth)
    hdul = fits.HDUList([hdu])
    hdul.writeto('/home/landman/Projects/SmoothCal/figures/IM_Smooth.fits', overwrite=True)
    hdul.close()

    hdu = fits.PrimaryHDU(IR_Smooth)
    hdul = fits.HDUList([hdu])
    hdul.writeto('/home/landman/Projects/SmoothCal/figures/IR_Smooth.fits', overwrite=True)
    hdul.close()

    # reweight assuming iid Gaussian errors
    Sigmapq = np.ones_like(Wpq_target)*theta0[-1]**2 # this is the given variance on the visibilities
    Wpq_reweighted = np.zeros_like(Wpq_target)
    print "Reweighting"
    for i, pq in enumerate(iter(pqlist)):
        p = int(pq[0]) - 1
        q = int(pq[1]) - 1
        gpvar = np.diag(gcov_smooth[p])[I2]
        gqvar = np.diag(gcov_smooth[q])[I2]
        for j in xrange(N_target):
            #Wpq_reweighted[p, q, j] = 1.0/np.abs(gpvar[j]/(gptemp[j]*gptemp[j].conj()) + gqvar[j]/(gqtempH[j]*gqtempH[j].conj())) #
            #print theta0[-1]**2, gpvar[j] + gqvar[j]
            Wpq_reweighted[p, q, j] = 1.0 / np.abs(theta0[-1]**2 + gpvar[j] + gqvar[j])
            Wpq_reweighted[q, p, j] = Wpq_reweighted[p, q, j]

    # make reweighted PSF
    print "Making reweighted PSF"
    PSF_weighted = np.zeros([2*Npix-1, 2*Npix-1])
    PSF_weighted_flat = PSF_weighted.flatten()
    for i, pq in enumerate(iter(pqlist)):
        p = int(pq[0]) - 1
        q = int(pq[1]) - 1
        uv = np.vstack((upq_target[i, :], vpq_target[i, :]))
        W = Wpq_reweighted[p, q, :]
        K = np.exp(-2.0j * np.pi * np.dot(lmPSF.T, uv.conj()))
        PSF_weighted_flat += np.dot(K, W).real

    PSF_weighted_max = PSF_weighted_flat.max()
    PSF_weighted_flat /= PSF_weighted_max
    PSF_weighted = PSF_weighted_flat.reshape(2*Npix-1, 2*Npix-1)

    hdu = fits.PrimaryHDU(PSF_weighted)
    hdul = fits.HDUList([hdu])
    hdul.writeto('/home/landman/Projects/SmoothCal/figures/PSF_weighted.fits', overwrite=True)
    hdul.close()

    # make reweighted dirty
    print "Making reweighted Dirty image"
    ID_weighted = np.zeros([Npix, Npix])
    ID_flat_weighted = ID_weighted.flatten()
    for i, pq in enumerate(iter(pqlist)):
        p = int(pq[0]) - 1
        q = int(pq[1]) - 1
        uv = np.vstack((upq_target[i, :], vpq_target[i, :]))
        V = Xpq_corrected_Smooth[p, q, :]*Wpq_reweighted[p, q, :]
        K = np.exp(-2.0j * np.pi * np.dot(lm.T, uv.conj()))
        ID_flat_weighted += np.dot(K, V).real
    ID_flat_weighted /= PSF_weighted_max
    ID_weighted = ID_flat_weighted.reshape(Npix, Npix)

    print "Cleaning"
    IM_weighted, IR_weighted = algos.Hogbom(ID_weighted, PSF_weighted, peak_fact=1.0e-3)

    # save images to fits
    hdu = fits.PrimaryHDU(ID_weighted)
    hdul = fits.HDUList([hdu])
    hdul.writeto('/home/landman/Projects/SmoothCal/figures/ID_weighted.fits', overwrite=True)
    hdul.close()

    hdu = fits.PrimaryHDU(IM_weighted)
    hdul = fits.HDUList([hdu])
    hdul.writeto('/home/landman/Projects/SmoothCal/figures/IM_weighted.fits', overwrite=True)
    hdul.close()

    hdu = fits.PrimaryHDU(IR_weighted)
    hdul = fits.HDUList([hdu])
    hdul.writeto('/home/landman/Projects/SmoothCal/figures/IR_weighted.fits', overwrite=True)
    hdul.close()

    # get SmoothCal model visibilities
    Xpq_pred = np.zeros_like(Xpq_target)
    Xpq_pred = R(IM_Smooth, upq_target, vpq_target, lm, pqlist, Xpq_pred)
    #Xpq_target2 = R(IM_target, upq_target, vpq_target, lm, pqlist, Xpq_pred)

    #print "Max diff = ", np.max(np.abs(Xpq_target - Xpq_target2))

    # stack target + calibrator fields
    Xpq_full = np.zeros([Na, Na, Nfull], dtype=np.complex)
    Xpq_full[:,:,I] = Xpq
    Xpq_full[:,:,I2] = Xpq_pred
    Vpq_full = np.zeros([Na, Na, Nfull], dtype=np.complex)
    Vpq_full[:,:,I] = Vpq
    Vpq_full[:,:,I2] = Vpq_target
    Wpq_full = np.zeros([Na, Na, Nfull])
    Wpq_full[:,:,I] = Wpq
    Wpq_full[:,:,I2] = Wpq_target


    # do SmoothCal on combined data sets
    gbar_full, gobs_full, Klist_full, Kylist_full, Dlist_full, theta = algos.SmoothCal(Na, Nfull, Xpq_full, Vpq_full,
                                                                                       Wpq_full, tfull, theta0, tol=1e-4,
                                                                                       maxiter=25, gbar=gmean_smooth)#,
                                                                                       #gobs=gbar_stef_lin_interp)
    # get StefCal model visibilities
    Xpq_pred_stef = np.zeros_like(Xpq_target)
    Xpq_pred_stef = R(IM_Stef, upq_target, vpq_target, lm, pqlist, Xpq_pred_stef)

    # do StefCal slef calibration
    gbar_stef_target, Sigmay_target = algos.StefCal(Na, N_target, Xpq_pred_stef, Vpq_target, Wpq_target, tp, tol=1e-4, maxiter=25) #Xpq_pred_stef
    gbar_stef_full = np.zeros([Na, Nfull], dtype=np.complex128)
    gbar_stef_full[:,I] = gbar_stef
    gbar_stef_full[:, I2] = gbar_stef_target
    Sigmay_full = np.zeros([Na, Nfull])
    Sigmay_full[:, I] = Sigmay.real
    Sigmay_full[:, I2] = Sigmay_target.real


    for i, pq in  enumerate(iter(pqlist)):
        p = int(pq[0])-1
        q = int(pq[1])-1

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 9))
        ax[0].fill_between(tfull, (gfull_true[p]*gfull_true[q].conj()).real + np.sqrt(1.0/Sigmay_full[p] + 1.0/Sigmay_full[q])/np.sqrt(2),
                           (gfull_true[p] * gfull_true[q].conj()).real - np.sqrt(1.0 / Sigmay_full[p] + 1.0 / Sigmay_full[q])/np.sqrt(2),
                           facecolor='b', alpha=0.25)
        ax[0].plot(tfull, (gfull_true[p]*gfull_true[q].conj()).real, 'k', label='True')
        ax[0].plot(tfull, (gbar_full[p]*gbar_full[q].conj()).real, 'b--', alpha=0.5, label='SmoothCal')
        ax[0].plot(tfull, (gbar_stef_full[p,:]*gbar_stef_full[q, :].conj()).real, 'g--', alpha=0.5, label='StefCal')
        #ax[0].errorbar(tfull, (gfull_true[0]*gfull_true[1].conj()).real, np.sqrt(1.0/Sigmay_full[0] + 1.0/Sigmay_full[1]), fmt='xr', alpha=0.25)
        ax[0].set_xlabel(r'$t$', fontsize=18)
        ax[0].set_ylabel(r'$Real(g_p g_q^\dagger)$', fontsize=18)
        #ax[0].legend()

        ax[1].fill_between(tfull, (gfull_true[p] * gfull_true[q].conj()).imag + np.sqrt(1.0 / Sigmay_full[p] + 1.0 / Sigmay_full[q])/np.sqrt(2),
                           (gfull_true[p] * gfull_true[q].conj()).imag - np.sqrt(1.0 / Sigmay_full[p] + 1.0 / Sigmay_full[q])/np.sqrt(2),
                           facecolor='b', alpha=0.25)
        ax[1].plot(tfull, (gfull_true[p] * gfull_true[q].conj()).imag, 'k', label='True')
        ax[1].plot(tfull, (gbar_full[p] * gbar_full[q].conj()).imag, 'b--', alpha=0.5, label='SmoothCal')
        ax[1].plot(tfull, (gbar_stef_full[p, :] * gbar_stef_full[q, :].conj()).imag, 'g--', alpha=0.5, label='StefCal')
        #ax[1].errorbar(tfull, (gfull_true[0] * gfull_true[1].conj()).imag, np.sqrt(1.0/Sigmay_full[0] + 1.0/Sigmay_full[1]), fmt='xr', alpha=0.25)
        ax[1].set_xlabel(r'$t$', fontsize=18)
        ax[1].set_ylabel(r'$Imag(g_p g_q^\dagger)$', fontsize=18)
        ax[1].legend(loc=2)

        fig.savefig('/home/landman/Projects/SmoothCal/figures/Full_sim_combined'+str(p)+str(q) +'.png', dpi = 250)

        # plot errors
        plt.figure('error2')
        plt.plot(tfull, np.abs(gfull_true[p] * gfull_true[q].conj() - gbar_full[p] * gbar_full[q].conj()), 'k.', label='SmoothCal')
        plt.plot(tfull, np.abs(gfull_true[p, :] * gfull_true[q, :].conj() - gbar_stef_full[p, :] * gbar_stef_full[q, :].conj()), 'g--', label='StefCal')
        plt.fill_between(tfull, np.sqrt(np.diag(Dlist_full[p].val).real + np.diag(Dlist_full[q].val).real), np.zeros(Nfull), facecolor='b', alpha=0.5)
        plt.xlabel(r'$t$', fontsize=18)
        plt.ylabel(r'$|\epsilon|$', fontsize=18)
        plt.legend()
        plt.savefig('/home/landman/Projects/SmoothCal/figures/Sim_error_combined'+str(p)+str(q) +'.png', dpi = 250)

        #plt.show()
        plt.close('all')

    # sim with real part of SelfCal and imaginary part from SmoothCal
    print "Applying gains"
    Xpq_corrected_Stef_Smooth = np.zeros_like(Xpq_target)
    for i, pq in enumerate(iter(pqlist)):
        p = int(pq[0])-1
        q = int(pq[1])-1
        gptemp = gbar_full[p, I2].real + gmean_smooth[p, I2].imag
        gqtempH = gbar_full[q, I2].real - gmean_smooth[q, I2].imag
        for j in xrange(N_target):
            Xpq_corrected_Stef_Smooth[p, q, j] = Vpq_target[p, q, j]/(gptemp[j]*gqtempH[j])
            Xpq_corrected_Stef_Smooth[q, p, j] = Xpq_corrected_Stef_Smooth[p, q, j].conj()

    # make dirty image
    # image corrected vis
    ID_Stef_Smooth = np.zeros([Npix, Npix])
    ID_flat_Stef_Smooth = ID_Stef_Smooth.flatten()
    print "Making Stef-SmoothCal Dirty"
    for i, pq in enumerate(iter(pqlist)):
        p = int(pq[0]) - 1
        q = int(pq[1]) - 1
        uv = np.vstack((upq_target[i, :], vpq_target[i, :]))
        V = Xpq_corrected_Stef_Smooth[p, q, :]
        K = np.exp(-2.0j * np.pi * np.dot(lm.T, uv.conj()))
        ID_flat_Stef_Smooth += np.dot(K, V).real
    ID_flat_Stef_Smooth /= PSFmax
    ID_Stef_Smooth = ID_flat_Stef_Smooth.reshape(Npix, Npix)

    print "Cleaning"
    IM_Stef_Smooth, IR_Stef_Smooth = algos.Hogbom(ID_Stef_Smooth, PSF, peak_fact=1.0e-3)

    # save images to fits
    hdu = fits.PrimaryHDU(ID_Stef_Smooth)
    hdul = fits.HDUList([hdu])
    hdul.writeto('/home/landman/Projects/SmoothCal/figures/ID_Stef_Smooth.fits', overwrite=True)
    hdul.close()

    hdu = fits.PrimaryHDU(IM_Stef_Smooth)
    hdul = fits.HDUList([hdu])
    hdul.writeto('/home/landman/Projects/SmoothCal/figures/IM_Stef_Smooth.fits', overwrite=True)
    hdul.close()

    hdu = fits.PrimaryHDU(IR_Stef_Smooth)
    hdul = fits.HDUList([hdu])
    hdul.writeto('/home/landman/Projects/SmoothCal/figures/IR_Stef_Smooth.fits', overwrite=True)
    hdul.close()