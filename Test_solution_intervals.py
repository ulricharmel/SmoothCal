"""
Test if solution interval idea is going to work. The idea is as follows:
 
 1) Do StefCal on Calibrator field with minimum solution interval
 2) Interpolate across target field by performing latent GPR on StefCal solutions
 3) Deconvolve resulting corrected data to get an estimate of model visibilities
 4) Do line search (for now) to find solution interval that minimises sum(Sigma_y - diag(gcov)) where Sigma_y is the 
    inverse of the maximum likelihood solution, gcov is the posterior covariance of the gains from GPR and the sum is 
    over all data
 5) Do StefCal with optimal solution interval
 6) Final deconvolution
"""


import numpy as np
import algorithms as algos
import itertools as it
from utils import apply_gains, R, RH, plot_fits
from GP.tools import draw_samples
from GP import temporal_GP
from astropy.io import fits
import matplotlib.pyplot as plt

if __name__=="__main__":
    # set save path
    savepath = '/home/landman/Projects/SmoothCal/figures/'
    # set the data
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
            I = np.append(I, np.arange(i*interval,(i+1)*interval)) #time indices of calibrator field
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
    lGP = 0.35
    sigmaf = 0.05
    sigman = 1.0
    theta0 = np.array([sigmaf, lGP, sigman])
    Nhypers = theta0.size

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
    IM[Npix//2, Npix//2] = 15.0
    IM[Npix//4, Npix//4] = 7.5
    IM[3*Npix//4, 3*Npix//4] = 5.0
    IM[Npix//4, 3*Npix//4] = 2.5
    IM[3*Npix//4, Npix//4] = 1.0
    IMflat = IM.flatten()

    # make sky model for imaging (target field should have low SNR)
    IM_target = np.zeros([Npix, Npix])
    IM_target[Npix//2, Npix//2] = 5.0

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

    plt.savefig(savepath + 'uv_coverage.png', dpi=250)

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
    Wpq = np.ones_like(Vpq, dtype=np.float64)/np.sqrt(2.0)

    # do StefCal cycle
    gbar_stef, JHJ = algos.StefCal(Na, Nt, Xpq, Vpq, Wpq, t, tol=5.0e-2, maxiter=25)
    Sigmay = np.sqrt(1.0/JHJ)

    # interpolate using StefCal data
    meanval = np.mean(gbar_stef, axis=1)
    GPlist = []
    thetaf = []
    theta0[0] = np.sqrt(2.0)*sigmaf
    theta0[-1] = np.sqrt(2.0)*sigman
    gsmooth = np.zeros_like(g_target)
    for i in xrange(Na):
        print "GPR on antenna ", str(i)
        # define mean function for antenna i
        meanf = lambda x: np.ones(x.size)*meanval[i]
        # initialise GP object for antenna i
        GPobj = temporal_GP.TemporalGP(t, tfull, gbar_stef[i, :], Sigmay=Sigmay[i], prior_mean=meanf,
                                          covariance_function='sqexp', mode="Full", M=25, L=20)
        # train GP for antenna i
        thetafi = GPobj.train(theta0)
        print thetafi
        thetaf.append(thetafi)  # need to reuse this later to reset the posterior
        GPobj.set_posterior(thetafi)
        GPlist.append(GPobj)
        meani = GPobj.post_mean.squeeze()
        gsmooth[i, :] = meani[I2]
        uncerti = np.sqrt(np.diag(GPobj.post_cov))

        # plt.figure('i')
        # plt.fill_between(tfull, meani.real + uncerti.real, meani.real - uncerti.real, facecolor='b', alpha=0.4)
        # plt.plot(tfull, meani.real, 'k')
        # plt.errorbar(t, gbar_stef[i, :].real, Sigmay[i], fmt='xr', alpha=0.5)
        # plt.show()

    # apply interpolated gains
    Vpq_corrected = np.zeros_like(Vpq_target)
    Vpq_corrected = apply_gains(Vpq_target, gsmooth, pqlist, N_target, Vpq_corrected)

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
        uv = np.vstack((upq_target[i, :], vpq_target[i, :]))
        W = Wpq_target[p, q, :]
        K = np.exp(-2.0j * np.pi * np.dot(lmPSF.T, uv.conj()))
        PSFflat += np.dot(K, W).real

    PSFmax = PSFflat.max()
    PSFflat /= PSFmax
    PSF = PSFflat.reshape(2*Npix-1, 2*Npix-1)

    hdu = fits.PrimaryHDU(PSF)
    hdul = fits.HDUList([hdu])
    hdul.writeto('/home/landman/Projects/SmoothCal/figures/PSF_sol_int.fits', overwrite=True)
    hdul.close()

    # image corrected data
    print "Making perfect Dirty"
    ID = np.zeros([Npix, Npix])
    ID = RH(Xpq_target, Wpq_target, upq_target, vpq_target, lm, ID, pqlist, PSFmax)

    # do the initial deconvolution
    print "Cleaning initial corrected data"
    IM, IR = algos.Hogbom(ID, PSF, peak_fact=1e-1)

    # plot deconvolution result
    plot_fits(IM, IR, ID, '1GC')

    # get JHJ

    # compute sum of differences

