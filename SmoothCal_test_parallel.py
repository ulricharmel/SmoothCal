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
from GP import temporal_GP

def train_ML(*args, **kwargs):
    try:
        return train_ML_impl(*args, **kwargs)
    except Exception as e:
        traceback_str = traceback.format_exc(e)
        raise StandardError("Error occurred. Original traceback "
                            "is\n%s\n" % traceback_str)

def train_ML_impl(theta0, gML, Sigmay, t, i):
    gmean = np.mean(gML)

    def gmeanf(t):
        return np.ones(t.size, dtype=np.complex128)*gmean

    GP = temporal_GP.TemporalGP(t, t, gML, Sigmay, prior_mean=gmeanf, covariance_function='mattern')

    bnds = ((1e-6, None), (1e-6, None), (1e-4, None))

    thetaf = GP.train(theta0, bounds=bnds)

    return thetaf, i


def get_hypers(theta0, gML, Sigmay, t):
    print "Optimising hypers on ML solution"
    futures = []
    Na = gML.shape[0]
    thetas = np.zeros([Na, theta0.size])
    max_jobs = np.min(np.array([psutil.cpu_count(logical=False), Na]))
    with cf.ProcessPoolExecutor(max_workers=max_jobs) as executor:
        for i in xrange(Na):
            future = executor.submit(train_ML, theta0.copy(), gML[i], Sigmay[i], t, i)
            futures.append(future)
        for f in cf.as_completed(futures):
            thetaf, i = f.result()
            thetas[i] = thetaf
            print "Completed ", i
    return thetas

if __name__=="__main__":
    # open MS
    ms = pt("/home/landman/Projects/Data/MS_dir/PKS1934/flagged.ms", readonly=True)

    # get time stamps
    times = ms.getcol("TIME")
    t = np.unique(times)
    Nt = t.size

    # scale and shift times
    tnorm = (t - np.min(t))/(np.max(t)-np.min(t)) - 0.5 # now tnorm lies between -0.5 and 0.5
    delt = tnorm[1] - tnorm[0]

    Np = (np.max(tnorm) - np.min(tnorm))//delt + 1
    tfull = tnorm[0] + np.arange(Np) * delt

    #print np.max(tfull), delt, Np, np.min(tnorm), np.max(tnorm), np.min(t)

    # get antennae list
    anta = ms.getcol("ANTENNA1")
    antb = ms.getcol("ANTENNA2")
    # number of antennae
    Na = antb[-1]+1

    # get single channel/correlation data
    ch = 2500
    corr = 0
    #mod_vis = ms.getcol("MODEL_DATA")[:, ch, corr]
    vis = ms.getcol("DATA")[:, ch, corr]
    mod_vis = np.ones_like(vis)

    # get flags
    flags = ms.getcol("FLAG")[:, ch, corr]

    # get weights
    weights = ms.getcol("WEIGHT")[:,0]

    # amplitude of calibrator taken from PKS document
    Flux = 15.0578061029
    mod_vis *= Flux

    ms.close()

    # set initial guess for covariance params
    lGP = 0.65
    sigmaf = 0.7
    sigman = 0.025
    theta0 = np.array([sigmaf, lGP, sigman])
    Nhypers = theta0.size

    # build cubes
    Xpq = np.zeros([Na, Na, Nt], dtype=np.complex128) # model vis cube
    Vpq = np.zeros([Na, Na, Nt], dtype=np.complex128) # vis cube
    Wpq = np.zeros([Na, Na, Nt], dtype=np.float64) # weights cube
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
            p = ant1[j]
            q = ant2[j]
            if not f:
                Xpq[p, q, i] = mvis[j]
                Xpq[q, p, i] = mvis[j].conj()
                Vpq[p, q, i] = dvis[j]
                Vpq[q, p, i] = dvis[j].conj()
                if (W[j] != 0.0):
                    Wpq[p, q, i] = W[j]
                    Wpq[q, p, i] = W[j]
                else:
                    Wpq[p, q, i] = 0.0
                    Wpq[q, p, i] = 0.0

    # create masked arrays
    # Xpq = ma.masked_equal(Xpq, np.nan)
    # Vpq = ma.masked_equal(Vpq, np.nan)
    # Wpq = ma.masked_equal(Wpq, np.nan)
    Xpq = ma.masked_equal(Xpq, 0.0)
    Vpq = ma.masked_equal(Vpq, 0.0)
    Wpq = ma.masked_equal(Wpq, 0.0)

    # get max lik solution
    gML, SigmayML = algos.StefCal(Na, Nt, Xpq, Vpq, Wpq, t, tol=1e-3, maxiter=25)

    # train GP on ML solution
    #thetas = get_hypers(theta0, gML, SigmayML, tnorm)

    #print thetas


    # run Smoothcal cycle
    #theta0[0] = np.sqrt(2)*sigmaf
    #theta0[-1] = np.sqrt(2) * sigman
    gbar, gobs, Klist, Kylist, Dlist, theta = algos.SmoothCal(Na, Nt, Xpq, Vpq, Wpq, tnorm, theta0, tol=1e-3, maxiter=25)
    #gbar, gobs, Klist, Kylist, Dlist, theta = algos.SmoothCal(Na, Nt, Xpq, Vpq, Wpq, tnorm, theta0, thetas=thetas, tol=1e-3, maxiter=25)

    #print gobs[0]

    # #print (theta==thetas).all()
    # for i in xrange(Na):
    #     Kylist[i].update(Klist[i], SigmayML[i])

    # Do interpolation
    meanval = np.mean(gbar, axis=1)
    #Np = 500
    #full = tnorm[0] + Np*delt ## .linspace(np.min(tnorm), np.max(tnorm), Np)

    gp, gcov = algos.get_interp(theta, tfull, meanval, gobs, Klist, Kylist, Dlist, Na)

    algos.plot_gains(gp, gML, tnorm, tfull, gcov)

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





