'''popmodule module to simulate absorption cross section spectrum.

Spectrum can be made up of one or more lines.
'''
from __future__ import division
# modules within package
from . import ohcalcs as oh
from . import absprofile as ab

# other modules
import numpy as np
import scipy.special
import pandas as pd
from scipy.constants import k
from scipy.constants import c, pi
import logging

LOGGER = logging.getLogger('popmodel.simspec')

def simline(hitline, xarr=None, press=oh.OP_PRESS, T=oh.TEMP, mass=oh.MASS):
    '''Calculate simulated absorption spectrum for a single HITRAN line.

    Accounts for relative population of lower state, assuming the state is
    a rotational state within OH X(v"=0), using an empirically-derived
    temperature-dependent partition function (see `oh_v0_rot_popdens`
    documentation). Absolute scaling will be incorrect if used for any other
    application.

    Parameters
    ----------
    hitline : ndarray
        1D recarray in format of a single line of output from
        loadhitran.processhitran.
    xarr : ndarray (optional)
        1D array of frequency values to calculate spectrum over, Hz. Default is
        None, and if no xarr is given, function creates one centered on the
        line with a fixed default width and resolution.
    press : float (optional)
        Pressure, torr. Default to value in ohcalcs.
    T : float (optional)
        Temperature, K. Default to value in ohcalcs.
    mass : float (optional)
        Mass, kg molec^-1. Default to value in ohcalcs.

    Returns
    -------
    lineseries : pd.Series
        Index of frequency values used to create spectrum, Hz. Values of
        effective absorption cross section values, cm^2.

    Notes
    -----
    Wrapper around `oh.voigt` and `sigma_eff`.

    '''
    # extract values from hitline
    E_low = hitline['E_low']*1.986455684e-23 # lower-state E, J (from cm^-1)
    g_air = hitline['g_air'] # air-broadening, HWHM at 296 K, cm-1 atm^-1
    wnum = hitline['wnum_ab'] # cm^-1
    Ja = hitline['Ja'] # total angular momentum, lower state
    Jb = hitline['Jb'] # total angular momentum, upper state
    Aba = hitline['Aba'] # s^-1

    freq = wnum*c*100 # Hz

    # Voigt line parameters
    sigma = oh.doppler_sigma(freq, T, mass)
    gamma = oh.pressure_gamma(g_air, press)

    # come up with xarr values if none passed to function
    if xarr == None:
        wc = gamma # lorentzian half-width
        wd = sigma * np.sqrt(2*np.log(2)) # doppler half-width, Wikipedia
        wv = wc/2+np.sqrt(wc**2/4+wd**2) # voigt hw approx of Whiting 1968
        width = 40 * wv # 40 of the approximate half-widths
        # hard-code 1000 points
        xarr = np.linspace(freq-width, freq+width, 1.e3)

    lineshape = oh.voigt(xarr, 1., freq, sigma, gamma, True)

    popdens = oh_v0_rot_popdens(Ja, T, E_low)
    stot = sigma_tot(wnum, Ja, Jb, Aba)

    # NB sigma_eff handles unit conversion from sigma_tot (cm^2 cm^-1) to
    # sigma_eff (cm^2 as function of Hz)
    seff = sigma_eff(popdens, stot, lineshape)

    lineseries = pd.Series(seff, index=xarr)
    # cut off first and last entry to zero to avoid interpolation artefacts
    # within simspec()
    lineseries.iloc[0] = 0
    lineseries.iloc[-1] = 0
    lineseries.index.name = "Frequency, Hz"
    return lineseries

def simspec(hitlines, press=oh.OP_PRESS, T=oh.TEMP):
    '''Combine set of hitlines into spectrum, as pandas DataFrame.

    Parameters
    ----------
    hitlines : ndarray
        recarray in the format that loadhitran.processhitran spits out

    Returns
    -------
    xarr : ndarray
        1D array of frequency values the spectrum was calculated over, Hz
    sigma_eff_array : ndarray
        2D array of effective absorption cross-section values across xarr for
        each line in hitlines, cm^2

    Notes
    -----
    Bundle up each pair of frequency values and absorption cross sections
    returned by `simline` into one pandas DataFrame object. Interpolate up to
    5 consecutive NaN values in each line, which can arise from overlapping
    lines.

    The DataFrame index is a combination of all frequency values used for all
    the lines, but the DataFrame is very sparse, saving on memory.
    
    '''
    linedict = {}
    for line in hitlines:
        linedict.update({line.label: simline(line, None, press, T)})
    specdata = pd.DataFrame(linedict)
    specdata.interpolate(method='linear', limit=5, inplace=True)
    return specdata

def sigma_eff(popdens, sigma_tot, lineshape):
    '''Calculate effective cross-section spectrum, cm^1, as function of Hz.

    Parameters
    ----------
    popdens : float
        Relative population in lower state
    sigma_tot : float
        Integrated cross-section over wavenumber space, cm^2 cm^-1
    lineshape : np.ndarray
        Array of normalized population as a function of frequency, Hz

    Returns
    -------
    sigma_eff : np.ndarray
        Array of effective cross-section, cm^2, as a function of frequency, Hz

    Notes
    -----
    Follow treatment in Dorn et al., J Geophys Res 100 (D4), 7397-7409, 1995.
    Represent absorption cross section spectrum as a product of the total
    integrated absorption cross-section, relative population in initial state,
    and Voigt lineshape.

    '''
    sigma_eff = popdens * sigma_tot*c*100 * lineshape # Dorn et al, Eq 7
    return sigma_eff

def oh_v0_rot_popdens(j, T, E):
    '''Calculate population in OH X(v"=0) rotational state.

    Parameters
    ----------
    j : float
        X(v"=0) rotational level
    T : float
        temperature, K
    E : float
        Energy of rotational level, J

    Returns
    -------
    popdens : float
        Population density in given rotational state.

    Notes
    -----
    Uses parameterized partition function given in Dorn et al., of Goldman and
    Gillis (1981) data.

    '''
    Qrot = (1.42e-6)*T**2 + 0.1485*T - 4.1
    popdens = (2*j+1)/Qrot*np.exp(-E/(k*T))
    return popdens

def sigma_tot(wnum, j_low, j_up, A):
    '''Calculate integrated absorption cross-section, cm^2 cm^-1.

    Parameters
    ----------
    wnum : float
        Wavenumber of transition, cm^-1
    j_low : float
        lower state rotational level
    j_up : float
        upper state rotational level
    A : float
        Einstein A-coefficient, s^-1

    Returns
    -------
    sigma_tot : float
        Integrated absorption cross-section, cm^2 Hz
    
    Notes
    -----
    Use Eq 3 in Dorn et al. (JGR 1995)

    '''
    sigma_tot = (1/(8*pi*c*100*wnum**2)*(2*j_up+1)/(2*j_low+1)*A)
    return sigma_tot

def makeindexnm(specdata):
    '''Given spectrum with frequency index, make with wavelength (nm) index.

    '''
    nmindex = c/specdata.index.values*1e9
    specdata_nm = pd.DataFrame(data=specdata.values, index=nmindex,
                               columns=specdata.columns)
    specdata_nm.sort_index(inplace=True)
    return specdata_nm

def spectocsv(csvfile, specdata):
    '''Write given specdata DataFrame to a csv file.

    Parameters:
    -----------
    csvfile : str
        Desired filename/path of csv output

    specdata : DataFrame
        DataFrame containing spectrum

    '''
    specdata.to_csv(csvfile)

def csvtospec(csvfile):
    '''Return data saved to a CSV file as a DataFrame.

    Parameters
    ----------
    csvfile : str
        Filename/path of csv input

    Returns
    -------
    specdata : pd.DataFrame
        DataFrame containing spectrum
    '''
    specdata = pd.read_csv(csvfile, index_col=0)
    return specdata

def _dornvoigt(wc, wd, wnum):
    '''Use Eq 13 in Dorn et al. to calculate voigt profile.
    
    Not for production use, just for check of consistency with other
    calculation method.

    Parameters
    ----------
    wc : float
        Collision half width parameter (Lorentzian) for selected background gas
        and pressure, pm
    wd : float
        Half width parameter for Doppler-broadened spectral line, pm
    wnum : float
        Center of feature, cm^-1

    Returns
    -------
    xarr : ndarray
        x-values for calculated profile, pm
    F_norm : ndarray
        Voigt profile, normalized to integral of 1

    '''
    # work in wavelength rather than wavenumber
    wlength = 1e10/wnum # pm

    # calculate voigt half width wv from wc and wd
    wv = wc/2+np.sqrt(wc**2/4+wd**2) # approx of Whiting 1968, pm

    # set up array of wavelengths to calculate profile over
    width = 20. # pm
    xarr = np.linspace(wlength-width, wlength+width, 1.e3) # pm
    xstep = xarr[1]-xarr[0] # pm

    # delta is 'normalized wavelength' to Voigt width wv
    delta = (xarr - wlength)/(wv*2) # ratio of pm

    # calculate Voigt profile: terms are functions of delta, weighted by the
    # ratio wc/wv. F1 + F2 is first approx given in Whiting et al. F3 is
    # refinement with better agreement -- has fractional powers of negative
    # numbers. To avoid getting NaN, take absolute value of delta first
    F1 = (1-wc/wv) * np.exp(-4*np.log(2)*delta**2)
    F2 = (wc/wv) / (1+4*delta**2)
    F3 = (0.016*(1.-wc/wv)*(wc/wv) *
          (np.exp(-.04*np.abs(delta)**2.25)-10/(10+np.abs(delta)**2.25)))
    raw_voigt = F1 + F2 + F3

    # normalize so that area under F is zero:
    av = 1.065 + 0.447*wc/wv + 0.058*(wc/wv)**2 # lineshape factor approx, Dorn
    norm_factor = 1/(2*wv*av) # see Dorn Eq 10
    voigt_norm = raw_voigt*norm_factor
    areaplotted = voigt_norm.sum() * xstep
    if areaplotted < 0.9:
        LOGGER.warning('Warning: profile contains <90%% of total area: %3d',
                       areaplotted)
    return xarr, voigt_norm

def _voigt(xarr, amp, xcen, wc, wd, normalized):
    """
    Calculate Voigt profile in pyspeckit-style with dornvoigt input.

    Only intended for use for comparison to _dornvoigt.

    Parameters
    ----------
    xarr : np.ndarray
        The X values over which to compute the Voigt profile, pm
    amp : float
        Amplitude of the voigt profile
        if normalized = True, amp is the AREA
    xcen : float
        The X-offset of the profile
    wc : float
        Collision half width parameter (Lorentzian) for selected background gas
        and pressure, pm
    wd : float
        Half width parameter for Doppler-broadened spectral line, pm
    normalized : bool
        Determines whether "amp" refers to the area or the peak of the voigt
        profile

    Returns
    -------
    V : np.ndarray
        Voigt profile y values for xarr, either normalized or not.

    """
    # calculate gamma, HWHM of collisional Lorentzian
    gamma = wc # pm
    # calculate sigma, std dev of Doppler Gaussian, from Doppler HWHM, wd
    sigma = wd / np.sqrt(2*np.log(2)) # Wikipedia, Voigt profile, pm
    # z is argument passed to Faddeeva function
    z = ((xarr-xcen) + 1j*gamma) / (sigma * np.sqrt(2))
    # voigt profile is real part of Faddeeva function, wofz() in scipy
    V = amp * np.real(scipy.special.wofz(z))
    if normalized:
        return V / (sigma*np.sqrt(2*np.pi))
    else:
        return V
