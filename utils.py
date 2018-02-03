
from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
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
            #print uv.shape, lm.shape
            Xpq[p,q,j] = np.dot(K, IMflat)
            # if q != p:
            #     Xpq[q,p,j] = Xpq[p,q,j].conj()
    return Xpq


def R2(IM, upq, vpq, lm, pqlist, Xpq):  # not working!!!
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
    i=0
    for pq in pqlist:
        p = int(pq[0])-1
        q = int(pq[1])-1
        uv = np.vstack([upq[i], vpq[i]])
        #print uv.shape, lm.shape
        K = np.exp(-2.0j * np.pi * np.dot(uv.T, lm))
        # print K.shape, uv.shape, lm.shape, IMflat.shape
        Xpq[p, q] = np.dot(K, IMflat)
        Xpq[q, p] = Xpq[p, q].conj()
        i += 1
    return Xpq

@jit(nopython=True, nogil=True, cache=True)
def R_jit(IM, upq, vpq, lm, pqlist, Xpq):
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
    i = 0
    for pq in pqlist:
        p = pq[0]
        q = pq[1]
        for j in xrange(Nt):
            uv = np.array([upq[i,j], vpq[i,j]])
            K = np.exp(-2.0j*np.pi*np.dot(uv,lm))
            Xpq[p,q,j] = np.dot(K.real, IMflat) + 1.0j*np.dot(K.imag, IMflat)
            Xpq[q,p,j] = Xpq[p,q,j].real - 1.0j*Xpq[p,q,j].imag
        i += 1
    return Xpq

def RH(Xpq, Wpq, upq, vpq, lm, ID, pqlist, PSFmax=None):
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
        X = Xpq[p, q, :]*Wpq[p, q, :]
        K = np.exp(-2.0j * np.pi * np.dot(lm.T, uv))
        ID_flat += np.dot(K, X).real
        # if q != p:
        #     uv = np.vstack((-upq[i, :], -vpq[i, :]))
        #     X = Xpq[q, p, :]*Wpq[q, p, :]
        #     K = np.exp(-2.0j * np.pi * np.dot(lm.T, uv))
        #     ID_flat += np.dot(K, X).real
    ID = ID_flat.reshape(ID.shape)
    if PSFmax is not None:
        return ID/PSFmax
    else:
        return ID

def RH2(Xpq, Wpq, upq, vpq, lm, ID, pqlist, PSFmax=None):
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
    i = 0
    for pq in pqlist:
        p = pq[0]
        q = pq[1]
        uv = np.vstack((upq[i, :], vpq[i, :]))
        X = Xpq[p, q, :]*Wpq[p, q, :]
        K = np.exp(-2.0j * np.pi * np.dot(lm.T, uv))
        ID_flat += np.dot(K, X).real
        i += 1
    ID = ID_flat.reshape(ID.shape)
    if PSFmax is not None:
        return ID/PSFmax
    else:
        return ID

@jit(nopython=True)
def RH_jit(Xpq, Wpq, upq, vpq, lm, ID, pqlist):
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
    i = 0
    for pq in pqlist:
        p = pq[0]
        q = pq[1]
        uv = np.vstack((upq[i, :], vpq[i, :]))
        X = Xpq[p, q, :]*Wpq[p, q, :]
        K = np.exp(-2.0j * np.pi * np.dot(lm.T, uv))
        ID_flat += np.dot(K, X)
        i += 1
    return ID_flat.reshape(ID.shape)

def plot_vis(Xpq, Xpq_corrected, Xpq_corrected2, upq, vpq, p, q):
    # plot absolute value of visibilities as function of baseline length
    plt.figure('visabs')
    plt.plot(np.abs(upq[p,:] - vpq[q,:]), np.abs(Xpq[p,q,:]), 'k+', label='True vis')
    plt.plot(np.abs(upq[p, :] - vpq[q, :]), np.abs(Xpq_corrected[p, q, :]), 'b+', label='Corrected vis 1')
    plt.plot(np.abs(upq[p, :] - vpq[q, :]), np.abs(Xpq_corrected2[p, q, :]), 'g+', label='Corrected vis 2')
    plt.savefig('/home/landman/Projects/SmoothCal/figures/abs_vis_compare.png', dpi=250)
    # plot phase of visibilities as function of baseline length
    plt.figure('visphase')
    plt.plot(np.abs(upq[p,:] - vpq[q,:]), np.arctan(Xpq[p,q,:].imag/Xpq[p,q,:].real), 'k+', label='True vis')
    plt.plot(np.abs(upq[p, :] - vpq[q, :]), np.arctan(Xpq_corrected[p,q,:].imag/Xpq_corrected[p,q,:].real), 'b+', label='Corrected vis 1')
    plt.plot(np.abs(upq[p, :] - vpq[q, :]), np.arctan(Xpq_corrected2[p,q,:].imag/Xpq_corrected[p,q,:].real), 'g+', label='Corrected vis 2')
    plt.savefig('/home/landman/Projects/SmoothCal/figures/phase_vis_compare.png', dpi=250)
    return

def plot_fits(IM, IR, ID, name):
    # save images to fits
    hdu = fits.PrimaryHDU(ID)
    hdul = fits.HDUList([hdu])
    hdul.writeto('/home/landman/Projects/SmoothCal/figures/ID_' + name + '.fits', overwrite=True)
    hdul.close()

    hdu = fits.PrimaryHDU(IM)
    hdul = fits.HDUList([hdu])
    hdul.writeto('/home/landman/Projects/SmoothCal/figures/IM_' + name + '.fits', overwrite=True)
    hdul.close()

    hdu = fits.PrimaryHDU(IR)
    hdul = fits.HDUList([hdu])
    hdul.writeto('/home/landman/Projects/SmoothCal/figures/IR_' + name + '.fits', overwrite=True)
    hdul.close()
    return

def plot_gains(tfull, gfull_true, Sigmay_full, gbar_full, gbar_stef_full, pqlist):
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
    return

def apply_gains(Vpq, g, pqlist, Nt, Xpq):
    for i, pq in enumerate(iter(pqlist)):
        p = int(pq[0])-1
        q = int(pq[1])-1
        gptemp = g[p]
        gqtempH = g[q].conj()
        for j in xrange(Nt):
            Xpq[p, q, j] = Vpq[p, q, j]/(gptemp[j]*gqtempH[j])
            Xpq[q, p, j] = Xpq[p, q, j].conj()
    return Xpq


if __name__=="__main__":
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

    # Set time axis
    Nt = 1000
    t = np.linspace(-5.5, 5.5, Nt)

    # this is to create the pq iterator (only works for N<10 antennae)
    Na = 9
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

    pqtup = []
    for i, pq in enumerate(iter(pqlist)):
        pqtup.append((int(pq[0]) - 1, int(pq[1]) - 1))

    # choose random antennae locations
    u = 10*np.random.random(Na)
    v = 10*np.random.random(Na)

    # create calibration baselines with time axis
    upq = np.zeros([N, Nt])
    vpq = np.zeros([N, Nt])
    phi = np.linspace(0, np.pi, Nt) # to simulate earth rotation
    for i, pq in enumerate(iter(pqlist)):
        #print i, pq
        upq[i, 0] = u[int(pq[0])-1] - u[int(pq[1])-1]
        vpq[i, 0] = v[int(pq[0])-1] - v[int(pq[1])-1]
        for j in xrange(1, Nt):
            rottheta = np.array([[np.cos(phi[j]), -np.sin(phi[j])], [np.sin(phi[j]), np.cos(phi[j])]])
            upq[i, j], vpq[i, j] = np.dot(rottheta, np.array([upq[i, 0], vpq[i, 0]]))

    # array to store visibilities
    Xpq = np.zeros([Na, Na, Nt], dtype=np.complex)

    # set weights
    Wpq = np.ones_like(Xpq, dtype=np.float64)

    # test self adjointness
    rndm_vis = np.random.randn(Na, Na, Nt) + 1.0j*np.random.randn(Na, Na, Nt)
    for pq in pqtup:
        p = pq[0]
        q = pq[1]
        rndm_vis[q, p] = 0.0 #rndm_vis[p, q].conj()
    rndm_img = np.random.randn(Npix, Npix)

    vis_from_img = np.zeros_like(Xpq)
    vis_from_img = R(rndm_img, upq, vpq, lm, pqlist, vis_from_img)
    img_from_vis = np.zeros_like(IM)
    img_from_vis = RH(rndm_vis, Wpq, upq, vpq, lm, img_from_vis, pqlist)

    # take dot products
    tmp1 = vis_from_img.flatten().dot(rndm_vis.flatten())
    tmp2 = img_from_vis.flatten().dot(rndm_img.flatten())

    print tmp1 - tmp2


    # # compare R accuracy and speed
    # Xpq_test = np.zeros_like(Xpq)
    # Xpq_test2 = np.zeros_like(Xpq)
    # from datetime import datetime
    # start_time = datetime.now()
    # Xpq_test = R(IM, upq, vpq, lm, pqlist, Xpq_test)
    # time_elapsed = datetime.now() - start_time
    # print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
    #
    # from datetime import datetime
    # start_time = datetime.now()
    # Xpq_test2 = R_jit(IM, upq, vpq, lm, pqtup, Xpq_test2)
    # time_elapsed = datetime.now() - start_time
    # print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
    #
    # # start_time = datetime.now()
    # # Xpq_test2 = R_jit(IM, upq, vpq, lm, pqtup, Xpq_test2)
    # # time_elapsed = datetime.now() - start_time
    # # print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
    # #
    # # Xpq_test2 = np.zeros_like(Xpq)
    # # start_time = datetime.now()
    # # Xpq_test2 = R_jit(IM, upq, vpq, lm, pqtup, Xpq_test2)
    # # time_elapsed = datetime.now() - start_time
    # # print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
    #
    # print (Xpq_test - Xpq_test2).max()
    #
    # # compare RH accuracy and speed
    # ID_test1 = np.zeros_like(IM)
    # ID_test2 = np.zeros_like(IM)
    # start_time = datetime.now()
    # ID_test1 = RH(Xpq_test2, Wpq, upq, vpq, lm, ID_test1, pqlist)
    # time_elapsed = datetime.now() - start_time
    # print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
    #
    # start_time = datetime.now()
    # ID_test2 = RH(Xpq_test, Wpq, upq, vpq, lm, ID_test2, pqlist)
    # time_elapsed = datetime.now() - start_time
    # print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
    #
    # print (ID_test1 - ID_test2).max()
    #
    #
    # plt.figure('ID')
    # plt.imshow(ID_test1)
    # plt.colorbar()
    #
    # plt.figure('ID2')
    # plt.imshow(ID_test2)
    # plt.colorbar()
    #
    # plt.show()


