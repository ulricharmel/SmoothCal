#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 12:47:40 2017

@author: landman

Here I define all the operators required for implementing SmoothCal. Eventually,
depending on trade off between speed and memory, we should be able to define
all operators by only storing the vector t. This woul be slower than, for
example storing the matrix of differences tt_ij = t[i] - t[j] which would be
slower than storing also the covariance function K_ij = k(tt_ij) etc..
The exact way we do this in the end will depend on practical considerations.

Nifty would have been useful here but unfortunately it doesn't support
uneven grids. Also I am not sure if the power_operators corresponding to 
stationary and isotropic covariance functions will be useful here
because the time coordinate doesn't necessarily fall on a regular grid. 

"""

import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg
from scipy.sparse import diags
#import GP
#import nifty
import matplotlib.pyplot as plt
import time
#from nifty import trace_probing
import pickle


class K_operator(object):
    def __init__(self, t, theta0, solve_mode="full", M=25, L=1.0, jit=1e-4):
        """
        This is a LinearOperator representation of the covariance matrix
        Input:
            t           - inputs
            theta0      - the vector of hyperparameters
            do_inverse  - if True computes the Cholesky factor. If false inverse is computed with preconditioned conjugate gradient.
                          As a rule of thumb this should be False whenever N>1000
            M           - the number of basis functions to use for the preconditioner
            L           - the support of the basis functions
            jit         - jitter for numerical stability of inverse
        """
        # set number of data points
        self.N = t.size

        #self.shape = (self.N, self.N)
        self.Nhypers = theta0.size
        
        # set inputs
        self.t = t

        from GP.kernels import exponential_squared
        self.kernel = exponential_squared.sqexp()
        #from GP.kernels import mattern
        #self.kernel = mattern.mattern(p=2, D=1)

        # set covariance function and evaluate
        self.theta = theta0
        # set initial jitter factor (for numerical stability of inverse)
        self.jit = jit

        # set derivative
        #self.dcov_func = lambda theta, mode: self.kernel.dcov_func(theta, self.tt, self.val, mode=mode)

        self.solve_mode = solve_mode
        if solve_mode=="full":
            # create vector of differences
            self.tt = np.tile(self.t, (self.N, 1)).T - np.tile(self.t, (self.N, 1))

            # evaluate covaraince matrix
            self.val = self.kernel.cov_func(self.theta, self.tt, noise=False)
            #self.val2 = self.kernel.cov_func2(self.theta, self.tt, noise=False)

            # get cholesky decomp
            self.L = np.linalg.cholesky(self.val + self.jit * np.eye(self.N)) #add jitter for numerical stability
            
            # get log determinant (could use entropic trace estimator here)
            self.logdet = 2*np.sum(np.diag(self.L)).real
            
            # get the inverse
            self.Linv = np.linalg.inv(self.L)

        # elif solve_mode=="approx" or solve_mode=="RR":
        #     # set up RR basis
        #     from GP.tools import make_basis
        #     from GP.basisfuncs import rectangular
        #     self.M = np.array([M])
        #     self.L = np.array([L])
        #     self.Phi = make_basis.get_eigenvectors(self.t.reshape(self.N, 1), self.M, self.L, rectangular.phi)
        #     Lambda = make_basis.get_eigenvals(self.M, self.L, rectangular.Lambda)
        #     self.s = np.sqrt(Lambda)
        #     # get spectral density
        #     tmp = self.kernel.spectral_density(self.theta, self.s)
        #     self.Lambdainv = diags(1.0/tmp)
        #     self.Sigmainv = diags(np.ones(self.N)/self.jit)
        #
        #     self.Mop = LinearOperator((self.N, self.N), matvec=self.M_func)
        #
        # elif solve_mode=="debug":
        #     # get cholesky decomp
        #     self.L = np.linalg.cholesky(self.val + self.jit * np.eye(self.N))  # add jitter for numerical stability
        #
        #     # get log determinant (could use entropic trace estimator here)
        #     self.logdet = 2 * np.sum(np.diag(self.L)).real
        #
        #     # get the inverse
        #     self.Linv = np.linalg.inv(self.L)
        #
        #     self.valinv = np.dot(self.Linv.T, self.Linv)
        #
        #     # this is for the preconditioning
        #     from GP.tools import make_basis
        #     from GP.basisfuncs import rectangular
        #     self.M = np.array([M])
        #     self.L = np.array([L])
        #     self.Phi = make_basis.get_eigenvectors(self.t.reshape(self.N, 1), self.M, self.L, rectangular.phi)
        #     Lambda = make_basis.get_eigenvals(self.M, self.L, rectangular.Lambda)
        #     self.s = np.sqrt(Lambda)
        #     # get spectral density
        #     tmp = self.kernel.spectral_density(self.theta, self.s) # figure out where factor of 2.5 comes from!!!!!!!!!!!!!!!!!!!!!!!!
        #     self.Lambdainv = np.diag(1.0/tmp)
        #     self.Lambda = np.diag(tmp)
        #     self.Sigmainv = np.diag(np.ones(self.N)/self.jit)
        #
        #     self.val2 = self.Phi.dot(self.Lambda.dot(self.Phi.T))
        #
        #     Z = self.Lambdainv + self.Phi.T.dot(self.Sigmainv.dot(self.Phi))
        #     LZ = np.linalg.cholesky(Z)
        #     LZinv = np.linalg.inv(LZ)
        #     Zinv = np.dot(LZinv.T, LZinv)
        #     self.valinv2 = self.Sigmainv - self.Sigmainv.dot(self.Phi.dot(Zinv.dot(self.Phi.T.dot(self.Sigmainv))))
        #
        #     self.Mop = LinearOperator((self.N, self.N), matvec=self.M_func)
        else:
            raise NotImplementedError("Still working on faster solve_mode")
    #
    # def M_func(self, x):
    #     """
    #     This is the preconditioner for K. Currently uses expansion i.t.o. Laplacian eigenfuncs
    #     :param x: the vector to act on
    #     """
    #     Sigmainvx = self.Sigmainv.dot(x)
    #     PhiTSigmainvx = self.Phi.T.dot(Sigmainvx)
    #     tmp = self.Lambdainv + self.Phi.T.dot(self.Sigmainv.dot(self.Phi))
    #     #L = np.linalg.cholesky(tmp)
    #     rhs_vec = np.linalg.solve(tmp, PhiTSigmainvx)  # might be possible to do this better. Currently O(M^3)
    #     return Sigmainvx - self.Sigmainv.dot(self.Phi.dot(rhs_vec))

    def update(self, theta):
        self.theta = theta
        # set derivative
        #self.dcov_func = lambda theta, mode: self.kernel.dcov_func(theta, self.tt, self.val, mode=mode)
        if self.solve_mode=="full":
            # update covariance matrix
            self.val = self.kernel.cov_func(self.theta, self.tt, noise=False)

            # get cholesky decomp (the try except statement is here to protect against numerically unstable inversions)
            F = True
            F2 = False
            jitfactor = 1
            while F:
                try:
                    self.L = np.linalg.cholesky(self.val + jitfactor*self.jit * np.eye(self.N)) # need to add jitter for numerical stability of inverse
                    F = False
                except:
                    jitfactor *= 10
                    F = True
                    F2 = True
            if F2:
                print "Had to increase jitter"
            # get log determinant
            self.logdet = 2*np.sum(np.log(np.diag(self.L))).real 
           
            # get the inverse
            self.Linv = np.linalg.inv(self.L)
        # else:
        #     tmp = self.kernel.spectral_density(self.theta, self.s)*2.5 # figure out where factor of 2.5 comes from!!!!!!!!!!!!!!!!!!!!!!!!
        #     self.Lambdainv = diags(1.0/tmp)
        #     self.Mop = LinearOperator((self.N, self.N), matvec=self.M_func)
        else:
            raise NotImplementedError("Still working on it")
        
    def _dot(self, x):
        if self.solve_mode == "full":
            return np.dot(self.val, x)
        else:
            raise NotImplementedError("Still working on it")
        
    def _idot(self, x):
        if self.solve_mode=="full": #slow but accurate version
            return self.Linv.conj().T.dot(self.Linv.dot(x))
        # elif self.solve_mode == "approx":# fast but not so accurate version (use if N>1000)
        #     tmp = cg(self.val + self.jit * np.eye(self.N), x, tol=1e-10, M=self.Mop)
        #     if tmp[1] >0:
        #         print "Warning cg tol not achieved"
        #     return tmp[0]
        # elif self.solve_mode == "RR": # very fast but currently not working version
        #     return self.Mop(x)
        else:
            raise NotImplementedError("Still working on faster solve_mode")

    def _dotdtheta(self, x, theta, mode):
        if mode < self.Nhypers-1:
            return np.dot(self.kernel.dcov_func(theta, self.tt, self.val, mode=mode), x)
        else:
            print "Max value of mode is ", self.Nhypers-1
            return 0

    def _dtheta(self, theta, mode):
        if mode < self.Nhypers-1:
            return self.kernel.dcov_func(theta, self.tt, self.val, mode=mode)
        else:
            raise NotImplementedError("Mode argument not supported")

    def KinvdKdtheta(self, theta, mode):
        """
        Currently will only work for sqexp covariance function
        :return: 
        """
        if mode == 0:
            return 2 * np.eye(self.N) / theta[0]
        elif mode == 1:
            return self.tt ** 2 / theta[1] ** 3
        else:
            raise NotImplementedError("Mode argument not supported")


class Ky_operator(object):
    def __init__(self, K, Sigmay, solve_mode="full"):
        """
        The Ky operator
        Input:
            K       - covariance matrix operator
            Sigmay  - the diagonal vector of (Ad.Sigmainv.A)
            sigman  - noise variance
        """
        self.K = K
        #self.sigman = sigman
        #self.Sigmayinv = diags(1.0/Sigmay)
        self.Sigmayinv = np.diag(1.0 / Sigmay)
        #self.shape = self.K.shape
        #self.solve_mode = solve_mode

        if self.K.solve_mode == "full":
            self.val = self.K.val + self.Sigmayinv #.toarray()
            # get cholesky decomp
            self.L = np.linalg.cholesky(self.val)
            self.logdet = 2.0*np.sum(np.log(np.diag(self.L)))
            
            # get the inverse
            self.Linv = np.linalg.inv(self.L)


        # elif solve_mode == "approx" or solve_mode == "RR":
        #     self.Myop = LinearOperator((K.N, K.N), matvec=self.My_func)
        #
        # elif solve_mode=="debug":
        #     self.Sigmayinv = np.diag(1.0 / Sigmay)
        #     self.val = self.K.val + self.Sigmayinv
        #     # get cholesky decomp
        #     self.L = np.linalg.cholesky(self.K.val + self.Sigmayinv)  # add jitter for numerical stability
        #
        #     # get log determinant (could use entropic trace estimator here)
        #     self.logdet = 2 * np.sum(np.diag(self.L)).real
        #
        #     # get the inverse
        #     self.Linv = np.linalg.inv(self.L)
        #
        #     self.valinv = np.dot(self.Linv.T, self.Linv)
        #
        #     self.val2 = self.K.Phi.dot(self.K.Lambda.dot(self.K.Phi.T)) + self.Sigmayinv
        #
        #     Z = self.K.Lambdainv + self.K.Phi.T.dot(self.Sigmayinv.dot(self.K.Phi))
        #     LZ = np.linalg.cholesky(Z)
        #     LZinv = np.linalg.inv(LZ)
        #     Zinv = np.dot(LZinv.T, LZinv)
        #     self.valinv2 = self.Sigmayinv - self.Sigmayinv.dot(self.K.Phi.dot(Zinv.dot(self.K.Phi.T.dot(self.Sigmayinv))))
        #
        #     self.Myop = LinearOperator((K.N, K.N), matvec=self.My_func)
        else:
            raise NotImplementedError("Still working on faster solve_mode")


    # def My_func(self, x):
    #     """
    #     This is the preconditioner for K. Currently uses expansion i.t.o. Laplacian eigenfuncs
    #     :param x: the vector to act on
    #     """
    #     Sigmainvx = self.Sigmayinv.dot(x)
    #     PhiTSigmainvx = self.K.Phi.T.dot(Sigmainvx)
    #     tmp = self.K.Lambdainv + self.K.Phi.T.dot(self.Sigmayinv.dot(self.K.Phi))
    #     rhs_vec = np.linalg.solve(tmp, PhiTSigmainvx)  # might be possible to do this better. Currently O(M^3)
    #     return Sigmainvx - self.Sigmayinv.dot(self.K.Phi.dot(rhs_vec))

    def update(self, K, Sigmay):
        self.K = K
        #self.Sigmayinv = diags(1.0 / Sigmay)
        self.Sigmayinv = np.diag(1.0 / Sigmay)
        #self.sigman = sigman
        if self.K.solve_mode == "full":
            # get cholesky decomp
            self.val = self.K.val + self.Sigmayinv #.toarray()
            self.L = np.linalg.cholesky(self.val)

            # get log determinant (could use entropic trace estimator here)
            self.logdet = 2 * np.sum(np.diag(self.L)).real

            # get the inverse
            self.Linv = np.linalg.inv(self.L)

        # else:
        #     self.Myop = LinearOperator((self.K.N, self.K.N), matvec=self.My_func)
        else:
            raise NotImplementedError("Still working on faster solve_mode")

    def _dot(self, x):
        return self.K._dot(x) + self.Sigmayinv.dot(x)
        
    def _idot(self, x):
        if self.K.solve_mode=="full":
            return self.Linv.conj().T.dot(self.Linv.dot(x))
        # elif self.solve_mode == "approx":
        #     tmp = cg(self.K.val + self.Sigmayinv.toarray(), x, M=self.Myop, tol=1e-8)
        #     if tmp[1] > 0:
        #         print "Warning cg did not converge"
        #     return tmp[0]
        # elif self.solve_mode == "RR":
        #     return self.Myop(x)
        else:
            raise NotImplementedError("Still working on faster solve_mode")

    def _dtheta(self, theta, mode):  # returns the value of dKdtheta
        if mode < self.K.Nhypers-1:
            return self.K._dtheta(theta, mode)
        elif mode == self.K.Nhypers-1:
            return np.diag(2.0*self.Sigmayinv/theta[-1]) # sigman is always last element of theta
        else:
            raise NotImplementedError("Mode argument not supported")

    def _dotdtheta(self, x, theta, mode):  # operator for dKdtheta
        if mode < self.K.Nhypers-1:
            return self.K._dotdtheta(x, theta, mode)
        elif mode == self.K.Nhypers-1:
            return 2.0*self.Sigmayinv.dot(x)/theta[-1] # sigman is always last element of theta
        else:
            raise NotImplementedError("Mode argument not supported")

    def interp(self, tp, theta, gobs, gmean):
        if self.K.solve_mode == "full":
            # get matrix of differences
            from GP.tools import abs_diff
            ttp = abs_diff.abs_diff(self.K.t, tp)
            ttpp = abs_diff.abs_diff(tp, tp)
            # get covariance matrices
            Kp = self.K.kernel.cov_func(theta, ttp, noise=False)
            Kpp = self.K.kernel.cov_func(theta, ttpp, noise=False)
            # get the mean function
            gbar = np.ones(tp.size, dtype=np.complex128)*gmean + np.dot(Kp.T, self._idot(gobs-np.ones(self.K.N, dtype=np.complex128)*gmean))
            gcov = Kpp - Kp.T.dot(self._idot(Kp))
            return gbar, gcov
        else:
            raise NotImplementedError("Still working on faster solve_mode")

class D_operator(object):
    def __init__(self, K, Ky):
        """
        The D operator
        Input:
            K       - covariance matrix operator
            Ky      - Ky operator
            Sigmay  - the diagonal vector of (Ad.Sigmainv.A)
        """
        self.K = K
        self.Ky = Ky
        #self.Sigmay = diags(1.0/self.Ky.Sigmayinv.diagonal())
        self.Sigmay = np.diag(1.0 / np.diag(self.Ky.Sigmayinv))
        #self.shape = self.K.shape

    def update(self, K, Ky):
        self.K = K
        self.Ky = Ky
        #self.Sigmay = diags(1.0/self.Ky.Sigmayinv.diagonal())
        self.Sigmay = np.diag(1.0 / np.diag(self.Ky.Sigmayinv))
        self.val = self.K.val - self.K.val.dot(self.Ky._idot(self.K.val))  ##### remomber to check mode==full here

    def _dot(self, x):
        tmp = self.K._dot(x)
        return tmp - self.K._dot(self.Ky._idot(tmp))
        
    def _idot(self, x):  # try to avoid using this as K._idot is numerically unstable
        return self.Sigmay.dot(x) + self.K._idot(x)
        
    def _dotdtheta(self, x, mode):
        if mode < self.K.Nhypers-1:
            dKdthetax = self.K._dotdtheta(x, mode)
            dKdthetaKyinvKx = self.K._dotdtheta(self.Ky._idot(self.K._dot(x)), mode)
            return dKdthetax - dKdthetaKyinvKx + self.K._dot(self.Ky._idot(dKdthetaKyinvKx)) - self.K._dot(self.Ky._idot(dKdthetax))
        elif mode == self.K.Nhypers-1:
            return self.K._dot(self.Ky._idot(self.Ky._dotdtheta(self.Ky._idot(self.K._dot(x)), mode)))
        else:
            raise NotImplementedError("Mode argument not supported")

    def _logdet(self):
        if self.K.solve_mode=="full":
            self.val = self.K.val - self.K.val.dot(self.Ky._idot(self.K.val))
            F = True
            F2 = False
            jitfactor = 1.0
            while F:
                try:
                    self.L = np.linalg.cholesky(self.val + jitfactor * self.K.jit * np.eye(self.K.N))
                    F = False
                except:
                    F2 = True
                    jitfactor *= 10.0
                    F = True
            self.logdet = 2.0*np.sum(np.log(np.diag(self.L)))
            if F2:
                print "Jitter added"
            return self.logdet.real
        else:
            raise NotImplementedError("Still working on faster solve_mode")

    def interp(self, tp, theta, gbar):
        if self.K.solve_mode == "full":
            # get matrix of differences
            from GP.tools import abs_diff
            ttp = abs_diff.abs_diff(self.K.t, tp)
            ttpp = abs_diff.abs_diff(tp, tp)
            # get covariance matrices
            Kp = self.K.kernel.cov_func(theta, ttp, noise=False)
            Kpp = self.K.kernel.cov_func(theta, ttpp, noise=False)
            # get the mean function
            gmean = np.dot(Kp.T, self._idot(gbar))
            gcov = Kpp - Kp.T.dot(self._idot(Kp))
            return gmean, gcov
        else:
            raise NotImplementedError("Still working on faster solve_mode")

# class test_operator(nifty.operator):
#     def __init__(self, domain, target, func):
#         self.domain = domain
#         self.target = target
#         self.func = func
#         self.sym = True
#         self.uni = False
#         self.imp = True
#
#     def _multiply(self, x):
#         return self.func(x)


if __name__=="__main__":
    # set theta
    sigmaf = 2.0
    l = 1.0
    sigman = 0.01
    theta = np.array([sigmaf, l, sigman])
    
    # create some inputs
    N = 200
    t = np.linspace(-1.0, 1.0, N)
    
    # construct linear operator
    K = K_operator(t, theta, solve_mode="full", M=12, L=5.0, jit=1e-4)

    tmp = K._dot(t)

    #from scipy.special import

    #
    # ti = time.time()
    # t2 = K.valinv2.dot(tmp)
    # tf = time.time()
    # print tf-ti
    #
    # print np.max(np.abs(t-t2)), np.max(np.abs(K.valinv - K.valinv2))
    # tmp = np.abs(K.valinv - K.valinv2)
    # print np.argwhere(tmp==np.max(tmp))
    #
    plt.figure()
    #plt.plot(t, K.val[:,100] - K.val2[:,100])
    plt.plot(t, K.val[:, -1])
    plt.plot(t, K.val2[:, -1])
    plt.show()
    #
    # plt.figure('2')
    # plt.plot(t, K.valinv[:,0])
    # plt.plot(t, K.valinv2[:,0])
    # plt.show()
    #
    # Knew = np.dot(Kinv2, K.val)
    # print np.linalg.cond(Knew), np.linalg.cond(K.val)
    #
    # print np.diag(Knew)

    #sigman = 0.00001
    #Sigmay = sigman**2*np.abs(np.random.randn(N))

    #Ky = Ky_operator(K, Sigmay, solve_mode="full")
    #
    # # f = open('/home/landman/Projects/SmoothCal/pickles/Ky.dat', 'w+')
    # # pickle.dump(Ky, f)
    # # f.close()
    # #
    # # f = open('/home/landman/Projects/SmoothCal/pickles/Ky.dat', 'r')
    # # Ky2 = pickle.load(f)
    # # f.close()
    #
    #tmp = Ky._dot(t)

    #t2 = Ky.valinv2.dot(tmp)

    #print np.max(np.abs(t-t2))
    # KyinvK = Ky._idot(K.val)
    # Kyinv = Ky.Linv.conj().T.dot(Ky.Linv)
    # KyinvK = np.dot(Kyinv, K.val)
    # KKyinv = np.dot(K.val, Kyinv)
    # print np.max(np.abs(KKyinv - KyinvK))

    # plt.figure()
    # plt.plot(t, Ky.val[:,0])
    # plt.plot(t, Ky.val2[:,0])
    # plt.show()
    #
    # plt.figure('2')
    # plt.plot(t, K.valinv[:,0])
    # plt.plot(t, K.valinv2[:,0])
    # plt.show()
    #
    #
    #
    #
    # D = D_operator(K, Ky)
    #
    # # f = open('/home/landman/Projects/SmoothCal/pickles/D.dat', 'w+')
    # # pickle.dump(D, f)
    # # f.close()
    # #
    # # f = open('/home/landman/Projects/SmoothCal/pickles/D.dat', 'r')
    # # D2 = pickle.load(f)
    # # f.close()
    #
    # tmp = D._dot(t)
    #
    # t2 = D._idot(tmp)
    #
    # print np.max(np.abs(t-t2))
#
#     # test nifty trace probing
#     dom = nifty.point_space(N, datatype=np.float64)
#     tar = nifty.point_space(N, datatype=np.float64)
#
#     test_func = lambda x: K._matvec(K._matvec(x))
#
#     test_op = test_operator(dom, tar, test_func)
#
#     res = nifty.trace_probing(test_op)
#
#     print np.sum(test_op.hathat(domain=dom, ncpu=4, nrun=100, loop=True).val), np.sum(np.diag(K.val.dot(K.val)))
#
#
#
#
#
#
# #
# #
# #    # test it
# #    tt = np.tile(t, (N, 1)).T - np.tile(t, (N, 1))
# #
# #    K2 = cfunc(theta, tt)
# #
# #    tmp2 = np.dot(K2, t)
# #
# #
# #
