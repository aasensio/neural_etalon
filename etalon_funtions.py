# ============================ IMPORTS ====================================== #
 
import numpy as np

# Optimization
from numba import float64, jit
 
# ============================= CONFIG ====================================== #
 
pi = np.pi
 
A = 0

# ============================= AUXILIAR ==================================== #

def sec(x):
    return 1 / np.cos(x)

def alpha1(a,b,F):
    return 2*a*b**2*np.sqrt(F)

def der_alpha1(b,F):
    #Derivative respect to 'a'
    return 2 * b ** 2 * np.sqrt(F)

def alpha2(a,b,F):
    alpha1=2*a*b**2*np.sqrt(F)
    return 2*alpha1*np.sqrt(F+1)

def der_alpha2(a,b,F):
    #Derivative respect to 'a'
    return 4*b**2*np.sqrt(F*(F+1))

def gamma1(a, F):
    return np.sqrt(F) * np.sin(a)

def der_gamma1(a,F):
    return np.sqrt(F)*np.cos(a)

def gamma2(a,b,F):
    return np.sqrt(F) * np.sin(a * (1- b ** 2))

def der_gamma2(a,b,F):
    #Derivative respect to 'a'
    return np.sqrt(F) * (1-b) * np.cos(a * (1- b ** 2))

def gamma3(F):
    return np.sqrt(F / (F+1))

def gamma4(a,b,F):
    return np.tan(a / 2 * (1-b ** 2)) / np.sqrt(F + 1)
 
def der_gamma4(a,b,F):
    #Derivative respect to 'a'
    return (1 - b) / (2 * np.sqrt(F+1)) *sec(a / 2 * (1-b ** 2)) ** 2

def gamma5(a,F):
    return np.tan(a/2)/np.sqrt(F+1)

def der_gamma5(a,F):
    return 1 / (2 * np.sqrt(F+1)) * sec(a/2) **2


# ============================= ETALON FS =================================== #

def transm(R):
    """
    Tau parameter (transmittance of the etalon)
    """
    return (1 - A / (1 - R)) ** 2

def F(R):
    """
    F parameter of the etalon
    """
    return 4 * R / (1 - R) ** 2
 

def RealE(alpha1, gamma1, gamma2):
    
    Ere = (2 / alpha1) * (np.arctan(gamma1) - np.arctan(gamma2))
    
    return Ere

def imE(R, alpha2,gamma3,gamma4,gamma5):

    Eim=(2 / alpha2) * (1 + R) / (1 - R) * \
    (np.log(((1+gamma3) ** 2 + gamma4 ** 2) / ((1-gamma3) ** 2 + gamma4 ** 2)) - \
     np.log(((1+gamma3) ** 2 + gamma5 ** 2) / ((1-gamma3) ** 2 + gamma5 ** 2)))
    
    return Eim
    
def der_RealE(a, b, F, alpha1, gamma1, gamma2):

    A = 1 / a * (np.arctan(gamma2) - np.arctan(gamma1))
    
    B = der_gamma1(a, F) / (1 + gamma1 ** 2)
    
    C = der_gamma2(a, b, F) / (1 + gamma2 ** 2)
    
    return 2 / (alpha1) * (A + B - C)    
    
def der_ImE(R,a,b,F,alpha2,gamma3,gamma4,gamma5):
 
    #Derivatives of alpha and gammas
    alpha2prima = der_alpha2(a,b,F)
    gamma4prima = der_gamma4(a,b,F)
    gamma5prima = der_gamma5(a,F)
    
    #Denominators of derivative of ratio
    denom1=((1-gamma3)**2+gamma4**2)*((1+gamma3)**2+gamma4**2)
    denom2=((1-gamma3)**2+gamma5**2)*((1+gamma3)**2+gamma5**2)

    der1=-alpha2prima/alpha2**2\
    *np.log(((1+gamma3)**2+gamma4**2)/((1-gamma3)**2+gamma4**2))\
    -(1/alpha2)*8*gamma3*gamma4*gamma4prima/denom1

    der2=-alpha2prima/alpha2**2\
    *np.log(((1+gamma3)**2+gamma5**2)/((1-gamma3)**2+gamma5**2))\
    -(1/alpha2)*8*gamma3*gamma5*gamma5prima/denom2
    
    return 2 * (1 + R) / (1 - R) * (der1 - der2)
        
# ============================= NUMERICAL =================================== #

@jit(float64(float64,float64,float64,float64,float64,float64,float64,float64,float64,\
float64,float64,float64,float64,float64,float64,float64,float64, float64), nopython=True, cache=True)    
def H11pintr_(r,theta,tau,R,F,xi,eta,xi0,eta0,k,f,wvl,theta3, nin, fnum, d, Da, wvl0):
    
    """
    Real part of H11tildeprima. All callable functions must be
    defined here for @jit to work
    """
    ll=wvl0+wvl0/(16*nin**2*fnum**2) #Peak wavelength
    m = round(2* nin * d / wvl0) #Order of the resonance peak
    delta_h = ( m * ll - 2 * nin * d)/ (2 * nin) #Variation of thickness to tune again to wvl0
    h = (d + delta_h) * Da
        
    ne = nin
    no = nin

    alpha=(xi-xi0)/f #cosine director in x direction
    beta=(eta-eta0)/f #cosine director in y direction
    def thetap(x,y,xi,eta,f):
        thetap=np.arccos(np.sqrt(f**2/((x-xi)**2+(y-eta)**2+f**2)))
        return thetap
    def H11(tau,R,F,wave,theta):
        dto=(4*np.pi*h/wave)*np.sqrt(no**2-1+(np.cos(theta))**2)
        H11=(np.sqrt(tau)/(1-R))*(1-R*np.exp(-1j*dto))*\
        np.exp(1j*dto/2)/(1+F*(np.sin(dto/2))**2)
        return H11
    def H22(tau,R,F,wave,theta,theta3):
        n=(no+ne)/2
        thetat=np.arcsin(np.sin(theta)/n)
        phi=(4*pi*h*n)*(ne-no)*\
        (np.sin(thetat-theta3))**2/(wave*np.sqrt(n**2-(np.sin(theta))**2))
        dto=(4*np.pi*h/wave)*np.sqrt(no**2-1+(np.cos(theta))**2)
        dte=phi+dto
        H22=(np.sqrt(tau)/(1-R))*(1-R*np.exp(-1j*dte))*\
        np.exp(1j*dte/2)/(1+F*(np.sin(dte/2))**2)
        return H22
    phip=np.arctan((eta0-r*np.sin(theta))/(xi0-r*np.cos(theta)))
    intr=r*np.real(\
    (H11(tau,R,F,wvl,\
    thetap(r*np.cos(theta),r*np.sin(theta),xi0,eta0,f))*np.cos(phip)**2\
    +H22(tau,R,F,wvl,thetap(r*np.cos(theta),r*np.sin(theta),xi0,eta0,f),theta3)*\
    np.sin(phip)**2)\
    *np.exp(-1j*k*(r*np.cos(theta)*alpha+r*np.sin(theta)*beta)))
    return intr
    
@jit(float64(float64,float64,float64,float64,float64,float64,float64,float64,float64,\
float64,float64,float64,float64,float64,float64,float64, float64, float64),nopython=True,cache=True)
def H11pinti_(r, theta, tau, R, F, xi, eta, xi0, eta0, k, f, wvl, theta3, nin, fnum, d, Da, wvl0):
    """
    Imaginary part of H11tildeprima. All callable functions must be
    defined here for @jit to work
    """

    ll=wvl0+wvl0/(16*nin**2*fnum**2) #Peak wavelength
    m = round(2* nin * d / wvl0) #Order of the resonance peak
    delta_h = ( m * ll - 2 * nin * d)/ (2 * nin) #Variation of thickness to tune again to wvl0
    h =(d + delta_h) * Da
    
    ne = nin
    no = nin

    alpha=(xi-xi0)/f #cosine director in x direction
    beta=(eta-eta0)/f #cosine director in y direction
    def thetap(x,y,xi,eta,f):
        thetap=np.arccos(np.sqrt(f**2/((x-xi)**2+(y-eta)**2+f**2)))
        return thetap
    def H11(tau,R,F,wave,theta):
        dto=(4*np.pi*h/wave)*np.sqrt(no**2-1+(np.cos(theta))**2)
        H11=(np.sqrt(tau)/(1-R))*(1-R*np.exp(-1j*dto))*\
        np.exp(1j*dto/2)/(1+F*(np.sin(dto/2))**2)
        return H11
    def H22(tau,R,F,wave,theta,theta3):
        n=(no+ne)/2
        thetat=np.arcsin(np.sin(theta)/n)
        phi=(4*pi*h*n)*(ne-no)*\
        (np.sin(thetat-theta3))**2/(wave*np.sqrt(n**2-(np.sin(theta))**2))
        dto=(4*np.pi*h/wave)*np.sqrt(no**2-1+(np.cos(theta))**2)
        dte=phi+dto
        H22=(np.sqrt(tau)/(1-R))*(1-R*np.exp(-1j*dte))*\
        np.exp(1j*dte/2)/(1+F*(np.sin(dte/2))**2)
        return H22
    phip=np.arctan((eta0-r*np.sin(theta))/(xi0-r*np.cos(theta)))
    inti=r*np.imag(\
    (H11(tau,R,F,wvl,thetap(r*np.cos(theta),r*np.sin(theta),xi0,eta0,f))*np.cos(phip)**2\
    +H22(tau,R,F,wvl,thetap(r*np.cos(theta),r*np.sin(theta),xi0,eta0,f),theta3)*\
    np.sin(phip)**2)\
    *np.exp(-1j*k*(r*np.cos(theta)*alpha+r*np.sin(theta)*beta)))
    return inti   
