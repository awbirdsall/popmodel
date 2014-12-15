'''simulate absorption cross section spectrum made up of one or more lines'''
# modules within package
import ohcalcs as oh
import atmcalcs as atm
import loadHITRAN as loadHITRAN

# other modules
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from scipy.constants import k
from scipy.constants import c, N_A, pi
from scipy.integrate import ode
from math import floor
import logging
import ConfigParser
logging.basicConfig(level=logging.INFO)
import pandas as pd

def dornvoigt(wc, wd, wnum):
    '''use Eq 13 in Dorn et al. to calculate voigt profile
    
    Parameters
    ----------
    wc : float
    Collision half width parameter for selected background gas and pressure, pm
    wd : float
    Half width parameter for Doppler-broadened spectral line, pm
    wnum : float
    Center of feature, cm^-1

    Outputs
    -------
    xarr : ndarray
    x-values for calculated profile, pm
    F_norm : ndarray
    Voigt profile, normalized to integral of 1
    '''

    wv = wc/2+np.sqrt(wc**2/4+wd**2) # approx of Whiting 1968, pm
    width = 20. # pm
    wlength = 1e10/wnum # pm
    xarr = np.linspace(wnum-width,wnum+width,1.e3) # pm
    xstep = xarr[1]-xarr[0] # pm
    # delta is 'normalized wavelength' to Voigt width wv
    delta = (xarr - wnum)/wv # ratio of pm
    # F1 + F2 is first approx given in Whiting et al. F3 is further refinement
    # with better agreement -- has fractional powers of negative numbers,
    # to avoid getting NaN, take absolute value of delta first
    F1 = (1-wc/wv)*np.exp(-4*np.log(2)*delta**2)
    F2 = (wc/wv)/(1+4*delta**2)
    F3 = 0.016*(1.-wc/wv)*(wc/wv)\
            *(np.exp(-.04*np.abs(delta)**2.25)-10/(10+np.abs(delta)**2.25))
    F = F1+F2+F3
    # normalize so that area under F is zero:
    av = 1.065 + 0.447*wc/wv + 0.058*(wc/wv)**2 # lineshape factor, Dorn
    norm_factor = 1/(wv*av) # see Dorn Eq 10
    F_norm = F*norm_factor
    return xarr,F_norm

def simline(hitline,xarr=None):
    '''Calculate simulated absorption spectrum for a single line.
    
    Follow treatment in Dorn et al., J Geophys Res 100 (D4), 7397-7409, 1995.
    Represent absorption cross section spectrum as a product of (1) the total
    integrated absorption cross-section sigma_tot, (2) relative population
    density popdens, and (3) voigt lineshape.
    '''
    # extract values from hitline
    E_low = hitline['E_low']*1.986455684e-23 # lower-state E, J (from cm^-1)
    g_air = hitline['g_air'] # air-broadening, HWHM at 296 K, cm-1 atm^-1
    wnum = hitline['wnum_ab'] # cm^-1
    Ja = hitline['Ja'] # total angular momentum, lower state
    Jb = hitline['Jb'] # total angular momentum, upper state
    Aba = hitline['Aba'] # s^-1
    # other parameters
    freq = wnum*c*100 # Hz
    T = 296 # K
    press = oh.op_press

    # (1) Calculate lineshape, in Hz domain, area normalized to 1:
    # Gaussian std dev for Doppler
    sigma = (k*T/(oh.mass*c**2))**(0.5) * freq

    # air-broadened HWHM at 296K, HITRAN (converted from cm^-1 atm^-1)
    # Could correct for temperature -- see Dorn et al. Eq 17
    gamma=(g_air*c*100) * press/760. # Lorentzian parameter

    # come up with xarr values if none passed to function
    width = 400e6
    if xarr == None:
        xarr = np.linspace(freq-width,freq+width,1.e3)

    lineshape = oh.voigt(xarr,1.,freq,sigma,gamma,True)

    # (2) calculate pop density
    # use HITRAN values for J, E_low; determine Qrot with parameterization
    # given in Dorn et al. of Goldman and Gillis (1981) data
    Qrot = (1.42e-6)*T**2 + 0.1485*T - 4.1
    popdens = (2*Ja+1)/Qrot*np.exp(-E_low/(k*T))

    # (3) total integrated absorption cross-section, cm^2
    # use Eq 3 in Dorn et al., using c in cm for result in cm^2
    sigma_tot=(1/(8*pi*c*100*wnum**2)*(2*Jb+1)/(2*Ja+1)*Aba) # cm^2
    # or Table 2.2 in Demtroeder p 41:
    # sigma_ij = (gj/gi)*c**2/(8*freq**2*d_freq) * Aji
    # sigma_tot = integrate sigma_ij d_freq
    # sigma_tot = (gj/gi)*c**2/(8*freq**2) * Aji
    # equivalent, with conversion wnum = freq / c

    # effective cross-section depends on population in lower state,
    # 'total' integrated cross-section, and lineshape (Voigt)
    sigma_eff = popdens * sigma_tot * lineshape # Dorn et al, Eq 7
    print "popdens, sigma_tot, lineshape \
    max:",popdens,sigma_tot,np.max(lineshape)
    return xarr,sigma_eff

def simspec(hitlines):
    '''combine set of hitlines into a spectrum'''
    # make unified x values to cover range around all center frequencies
    width = 400e6
    mincenterfreq = np.min(hitlines['wnum_ab'])*c*100 # Hz
    maxcenterfreq = np.max(hitlines['wnum_ab'])*c*100 # Hz
    xarr = np.linspace(mincenterfreq-width, maxcenterfreq+width, 1.e3)
    # make ndarray for resulting absorprtion cross section spectra
    sigma_eff_array = np.empty([hitlines.size,xarr.size])
    for index, line in enumerate(hitlines):
        freq,sigma_eff_array[index]=simline(line,xarr)
    return xarr,sigma_eff_array
