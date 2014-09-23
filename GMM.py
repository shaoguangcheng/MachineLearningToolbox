#!/usr/bin/env python

"""Gaussian Mixed Model"""

import numpy as np
import pylab as pl

class GMM(object) :
	def __init__(self, X, K_or_centroid) :
		"""
			X 			  : input data, size : N x D
			K_or_centroid : if the type of K_or_centroid is int, then it represents the number of Gaussian components
						    if the type of K_or_centroid is np.ndarray, then it represents the initial centers of Gaussian, size K X D
		"""
		(self.N, self.D) = X.shape
		self.X = X
		if isinstance(K_or_centroid, int) : # if number of Gaussian is specified
			self.K = K_or_centroid
			np.random.shuffle(X)
			centroid = X[:self.K, :]

		if isinstance(K_or_centroid, np.ndarray) : # if the initial centers of Gaussian are apecified
			self.K = K_or_centroid.shape[0]
			centroid = K_or_centroid;
		if 0 :
			self.miu = centroid; # initialize miu (K x D)
			self.pi = np.array([1.0/self.K]*self.K)
			self.sigma = [np.eye(self.D)]*self.K
			minDistindex = np.zeros((self.N))
			repeatArray = [self.K] * self.N
			XTmp = np.repeat(X, repeatArray, axis = 0)
			centroid = np.resize(centroid, [self.N * self.K, self.D])

			subTmp = XTmp - centroid
			distanceMat = np.sum(subTmp * subTmp, axis = 1).reshape((self.N, self.K))

			for i in range(self.N) :
				index = 0
				for j in range(self.K) :
					if distanceMat[i, j] < distanceMat[i, index] :
						index = j
				minDistindex[i] = index

			self.pi  = np.zeros((self.K)) # initialize pi (1 x K)
			self.sigma = [] # initialize sigma (D x D x K)
			for k in range(self.K) :
				self.sigma.append(np.zeros((self.D, self.D)))

			for index in range(self.N) :
				k = int(minDistindex[index])
				self.pi[k] = self.pi[k] + 1
				bias = X[index] - self.miu[k]
				self.sigma[k] = self.sigma[k] + np.dot(bias.transpose(), bias)

			for k in range(self.K) :
				self.sigma[k] = self.sigma[k]*1.0/self.pi[k]
				self.pi[k] = self.pi[k]/self.N

		self.miu = centroid
		self.pi  = np.array([1.0/self.K]*self.K)
		self.sigma = [np.eye(self.D)]*self.K

	def fit(self) :
		threshold = 1e-4

		Lpre = -1 * np.inf
		PGamma = np.zeros((self.K, self.N)) # Pik the probability of the i-th example from k-th component
		delta = np.eye(self.D)
		while True :
			for index in range(self.K) :
				XShift = self.X -self.miu[index, :]
				self.sigma[index] = self.sigma[index]+delta 
				invSigma = np.array(np.matrix(self.sigma[index]).I) # here must be careful, if sigma is singular, 
																		  # then we have to add a small positive delta to sigma's diagnol.
																		  # Delta can not too small, often we can select 0.1 or 0.01
				tmp = np.sum(np.dot(XShift, invSigma) * XShift, axis = 1)

				coef = (2*np.pi)**(-1.0*self.D/2) * (np.linalg.det(invSigma))**(0.5)
				PGamma[index, :] = self.pi[index] * coef * np.exp(-1*0.5*tmp)

			Pik = PGamma.transpose()
			Pik = Pik/np.sum(Pik, axis = 1).reshape((self.N,1))
			Nk = np.sum(Pik, axis = 0)
			self.pi = Nk/self.N # update pi
			for index in range(self.K) :
				self.miu[index, :] = np.sum((Pik[:, index].reshape(self.N, 1) * self.X), axis = 0)/Nk[index] #update miu
				XShift = self.X - self.miu[index, :]
				tmp = np.dot((Pik[:, index].reshape(self.N, 1)* XShift).transpose(), XShift) 
				self.sigma[index] = tmp/Nk[index] #update sigma

			L = np.sum(np.log(PGamma))
			if np.absolute(L - Lpre) < threshold :
				break
			Lpre = L

		return (self.pi, self.miu, self.sigma, Pik)

if __name__ == "__main__" :
	X = 0.1*np.random.randn(10000,2)
	X[:2000,] = X[:2000,] + 0.1*np.random.randn(2000,2)+3

	c = np.array([[0.3, 0.3],[13, 11]]) 

	gmm = GMM(X, c)
	gmm.fit()

	pl.figure()
	pl.scatter(X[:2000, 0], X[:2000, 1], c = 'r')
	pl.scatter(X[2000:, 0], X[2000:, 1], c = 'b')
	pl.show()

	print gmm.pi
	print gmm.miu
	print gmm.sigma[1]

