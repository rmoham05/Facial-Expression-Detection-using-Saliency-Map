
import numpy as np
import math
import skfmm
import cv2

import scipy

from scipy.spatial import distance

def grad(x):
    out = np.array(np.gradient(x))
    return out

def norm(x, axis=0):
    out = np.sqrt(np.sum(np.square(x), axis=axis))
    return out

# stopping function
def stopping_fun(x):
    return 1./((1. + norm(grad(x)))**2)

# initialize phi
def default_phi(x):
    phi = np.ones(x.shape) * -1
    '''
    y,x = x.shape
    cy = int(y * 0.75)
    cx = int(x/2)
    r = 5
    
    for i in range(x):
        for j in range(y):
            if ((i-cx)**2 + (j-cy)**2) <= r**2:
                phi[j,i] = 1.
    
    #phi[cy,cx] = -1
    phi = skfmm.distance(phi)
    #plt.imshow(np.clip(phi, 0, 255), cmap='gray')
    #plt.title('original phi')
    #plt.show()
    '''
    
    phi[5:-5, 5:-5] = 1.
    
    return phi

'''
continuous approx of impulse function
A: array
'''
def delta(A, eps=0.1):

    out = np.where(abs(A) < eps, 1/(2*eps) * (1 + np.cos(math.pi * A/eps)), 0)
    '''
    y,x = A.shape
    out = np.zeros(A.shape)

    for i in range(x):
        for j in range(y):
            if abs(A[j,i]) < eps:
                out[j,i] = 1/(2*eps) * (1 + np.cos(math.pi * A[j,i] / eps))
    '''
    return(out)

'''
continuous approx of step function
A: array
'''
def heavi(A, eps=0.5):
    
    out = np.where(A >= 0, 1, 0)
    '''
    y,x = A.shape
    out = np.zeros(A.shape)

    for i in range(x):
        for j in range(y):

            if A[j,i] > eps:
                out[j,i] = 1.
            elif abs(A[j,i]) <= eps:
                out[j,i] = 1./2. * (1. + A[j,i] / eps + 1./np.pi * np.sin(math.pi * A[j,i] / eps) )
    '''
    return out

def div(fx, fy):
    fyy, fyx = grad(fy)
    fxy, fxx = grad(fx)
    return fxx + fyy

''' 
Level Set Function
im: input image to segment
dE: derivative of energy, function of phi and im, 
phi0: intial phi (level set function)
maxiter: maximum number of iterations
'''
'''
parameters: dt, eps, v, mu, lamb1, lamb2, alpha
dt: step size
eps: epsilon for cts approx of delta/heavi
v, mu, lamb1, lamb2: parameters for energy function
alpha: width of gaussian smoothing
'''
def levelset(im, phi0, maxiter, prior, dt=1, v=0.1, mu=1, lamb1=2, lamb2=1, alpha=0.5, tau=1):

    im_smooth = scipy.ndimage.filters.gaussian_filter(im, alpha)
    
    prior_match = shape_match(phi0, prior)
    
    phi2 = phi0

    for i in range(maxiter):

        phi = phi2
        
        phi2 = phi + dt * dE(phi, im_smooth, prior_match, v, mu, lamb1, lamb2, tau) #* 1/(i + 1)
        
        try:
            phi2 = skfmm.distance(phi2)
        except:
            print('error in level set')
            phi2 = skfmm.distance(phi2 - np.mean(phi2))
        
        #plt.imshow(np.clip(phi2, 0, 255), cmap='gray')
        #plt.show()
        
        #print(np.sum(np.sum((phi - phi2)**2)))
        
        #if np.sum(np.sum((phi - phi2)**2)) <= 110000 and i > 10:
            #break
        
        
    return phi
    
# dphi/dt = delta(phi) * ( v * kappa(phi) - (I - c1)^2 + (I - c2)^2 - mu - tau * C(phi,phi0))
def dE(phi, im, prior_match, v=1000, mu=1000, lamb1=0.9, lamb2=1.1, tau=10):

    #v = 1
    #mu = 1

    fy, fx = grad(phi)
    norm = np.sqrt(fx**2 + fy**2)
    Nx = fx / (norm + 1e-8)
    Ny = fx / (norm + 1e-8)
    vkappa = v * div(Nx, Ny)

    deltaphi = delta(phi)
    heaviphi = np.heaviside(phi, 1)

    c1 = np.sum(np.sum(im * heaviphi / np.sum(np.sum(heaviphi))))
    #lamb1 = 1.

    c2 = np.sum(np.sum(im * (1 - heaviphi)) / np.sum(np.sum(1 - heaviphi)))
    #lamb2 = 1.
    '''
    print(
            str(np.sum(deltaphi*vkappa)) + ", " + 
            str(np.sum(deltaphi*lamb1*(im-c1)**2)) + ", " + 
            str(np.sum(deltaphi*lamb2*(im-c2)**2)) + ", " + 
            str(np.sum(deltaphi*mu)) + ", " + 
            str(np.sum(deltaphi*tau*prior_match)))
    '''
    return deltaphi * (vkappa - lamb1*(im - c1)**2 + lamb2*(im - c2)**2 - mu + tau * prior_match)

def shape_match(phi, prior):
    
    out = np.zeros(phi.shape)
    y,x = phi.shape
    
    contour, hierarchy = cv2.findContours(prior, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    for i in range(x):
        for j in range(y):
            out[j, i] = np.min(distance.cdist(np.array([[i,j]]), np.array(contour[0][:,0,:])))
            
    return out

