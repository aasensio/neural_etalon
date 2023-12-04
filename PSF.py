#%%
# ============================ IMPORTS ====================================== #
 
import numpy as np
import matplotlib.pyplot as plt

import time

# Integrals
from scipy.integrate import simps, nquad

# Own Libs
import etalon_funtions as etf
# import functions as ft

# ============================= CONFIG ====================================== #

# pi = np.pi

Et = {
      'R' : 0.892,
      'n' : 2.3268,
      'd' : 281e-6,
      'fnum' : 60
      }

# ============================================================================ #
#%%


"""
La función PSF depende de 5 parámetros importantes :

 - angle 1 : Proyeccion del ángulo en el eje X.
             Varía entre: 0 y 0.4
 - angle 2 : Proyeccion del ángulo en el eje Y.
             Varía entre: 0 y 0.4
 - xi  : Coordenada en eje x en metros.
         Varía entre +- 32 x tamaño del pixel (10 micras por ejemplo)
 - eta : Coordenada en eje y en metros
         Varía entre +- 32 x tamaño del pixel (10 micras por ejemplo)
 - Da  : Variación relativa del espesor del etalón.
         Varía entre 0.999999 y 1.000001

 - Et : Propiedades del etalón que no hace falta cambiar.

"""


def PSF(wvls, angle1, angle2, xi, eta, Da, Et):

    """
    Function that calculates the Etalon transmission profile.

    Parameters
    ----------
    angle1 : float
        Projection of the angle in X axis
    angle2 : float
        Projection of the angle in Y axis
    xi : float
        Coordinate 1 in detector plane in meters
    eta : float
        Coordinate 2 in detector plane in meters
    Da : float
        Relative thickness variation of etalon
    Et : dict
        Etalon Properties.

    Returns
    -------
    Inum : array
        Etalon Profile.

    """

    # Configuration    
    l0 = 6173.5 * 1E-10
    Nl = int(len(wvls))
    Inum = np.zeros(Nl)
    tau  = etf.transm(Et['R'])
    
    xi0  = np.sin(np.deg2rad(angle1) * Et['fnum'])
    eta0 = np.sin(np.deg2rad(angle2) * Et['fnum'])
    
    xi  += xi0
    eta += eta0
    
    theta3 = 0
    f = Et['fnum'] * 2 
    lims = [[0, 1], [0, 2 * np.pi]]  # Integral limits
    accur = {'limit': 50, 'epsabs':0,'epsrel': 1.49e-8}  # Optional arguments
    j = -1
    F = etf.F(Et['R'])

    # Loop over wavelengths. 
    for wvli in wvls:
        
        j += 1
        k = 2 * np.pi / wvli
        
        params = (tau, Et['R'], F, xi, eta, xi0, eta0, k, f, wvli, theta3, 
                  Et['n'], Et['fnum'], Et['d'], Da, l0)
        H11tr = nquad(etf.H11pintr_, lims, args=params, opts=accur)
        H11ti = nquad(etf.H11pinti_, lims, args=params, opts=accur)
        H11t = (H11tr[0] + 1j * H11ti[0]) / (np.pi * 1 ** 2)
        Inum[j] = np.real(H11t * np.conj(H11t))

    return Inum


if __name__ == '__main__':
    # Example of profile excution
    Wavelengths =  np.arange(6173, 6174, 0.05) * 1E-10
    tic = time.time()
    print(f"Computing etalon profile...")
    Etalon = PSF(Wavelengths, 0.3, 0.2, 0.0, 0, 1, Et)
    print(f"FINISHED in {round(time.time() - tic, 2)}s")


    plt.figure()
    plt.plot([6173.5* 1E-10, 6173.5* 1E-10], [0, 1])
    plt.plot(Wavelengths, Etalon, c = "m")
    plt.show()


    # %%
