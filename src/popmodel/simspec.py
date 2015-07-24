'''popmodule module to simulate absorption cross section spectrum.

Spectrum can be made up of one or more lines.
'''
# modules within package
from . import ohcalcs as oh

# other modules
import numpy as np
import scipy.special
import pandas as pd
from scipy.constants import k
from scipy.constants import c, pi
import logging

LOGGER = logging.getLogger('popmodel.simspec')

def dornvoigt(wc, wd, wnum):
    '''use Eq 13 in Dorn et al. to calculate voigt profile

    Parameters
    ----------
    wc : float
    Collision half width parameter (Lorentzian) for selected background gas and
    pressure, pm
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
    F3 = 0.016*(1.-wc/wv)*(wc/wv)\
            * (np.exp(-.04*np.abs(delta)**2.25)-10/(10+np.abs(delta)**2.25))
    raw_voigt = F1 + F2 + F3

    # normalize so that area under F is zero:
    av = 1.065 + 0.447*wc/wv + 0.058*(wc/wv)**2 # lineshape factor approx, Dorn
    norm_factor = 1/(2*wv*av) # see Dorn Eq 10
    voigt_norm = raw_voigt*norm_factor
    areaplotted = voigt_norm.sum() * xstep
    if areaplotted < 0.9:
        LOGGER.warning('Warning: profile contains <90% of total area: {}'
                       .format(areaplotted))
    return xarr, voigt_norm

def voigt(xarr, amp, xcen, wc, wd, normalized):
    """
    *modified version for quality control purposes -- comparison to dornvoigt*

    Normalized Voigt profile from pyspeckit, on Github.

    z = (x+i*gam)/(sig*sqrt(2))
    V(x,sig,gam) = Re(w(z))/(sig*sqrt(2*pi))

    The area of V in this definition is 1.  If normalized=False, then you can
    divide the integral of V by sigma*sqrt(2*pi) to get the area.

    Original implementation converted from
    http://mail.scipy.org/pipermail/scipy-user/2011-January/028327.html (had an
    incorrect normalization and strange treatment of the input parameters)

    Modified implementation taken from wikipedia, using the definition.
    http://en.wikipedia.org/wiki/Voigt_profile

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
    Collision half width parameter (Lorentzian) for selected background gas and
    pressure, pm
    wd : float
    Half width parameter for Doppler-broadened spectral line, pm
    normalized : bool
    Determines whether "amp" refers to the area or the peak of the voigt
    profile

    Outputs
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

def simline(hitline, xarr=None, press=oh.OP_PRESS, T=oh.TEMP):
    '''Calculate simulated absorption spectrum for a single HITRAN line.

    Follow treatment in Dorn et al., J Geophys Res 100 (D4), 7397-7409, 1995.
    Represent absorption cross section spectrum as a product of (1) the total
    integrated absorption cross-section sigma_tot, (2) relative population
    density popdens, and (3) voigt lineshape. To generate Voigt lineshape, use
    scipy wofz function as implemented by pyspeckit, instead of Dorn et al.'s
    use of older literature approximation of Whiting et al., J Quant Spectrosc
    Radiat Transfer 8, 1379-1384, 1968. (Checked two methods for
    near-consistency.)

    Parameters
    ----------
    hitline : ndarray
    1D recarray in format of a single line of output from
    loadhitran.processhitran.
    xarr : ndarray (optional)
    1D array of frequency values to calculate spectrum over, Hz. Default is
    None, and if no xarr is given, function creates one centered on the line
    with a fixed default width and resolution.
    press : float (optional)
    Pressure, torr. Default to value in ohcalcs.
    T : float (optional)
    Temperature, K. Default to value in ohcalcs.

    Outputs
    -------
    lineseries : pd.Series
    Index of frequency values used to create spectrum, Hz. Values of effective
    absorption cross section values, cm^2.
    '''
    # extract values from hitline
    E_low = hitline['E_low']*1.986455684e-23 # lower-state E, J (from cm^-1)
    g_air = hitline['g_air'] # air-broadening, HWHM at 296 K, cm-1 atm^-1
    wnum = hitline['wnum_ab'] # cm^-1
    Ja = hitline['Ja'] # total angular momentum, lower state
    Jb = hitline['Jb'] # total angular momentum, upper state
    Aba = hitline['Aba'] # s^-1

    # convert wnum to frequency
    freq = wnum*c*100 # Hz

    # (1) Calculate lineshape, in Hz domain, area normalized to 1:
    # Gaussian std dev for Doppler
    sigma = (k*T/(oh.MASS*c**2))**(0.5) * freq

    # air-broadened HWHM at 296K, HITRAN (converted from cm^-1 atm^-1)
    # Could correct for temperature -- see Dorn et al. Eq 17
    gamma = (g_air*c*100) * press/760. # Lorentzian parameter

    # come up with xarr values if none passed to function
    if xarr == None:
        wc = gamma # lorentzian half-width
        wd = sigma * np.sqrt(2*np.log(2)) # doppler half-width, Wikipedia
        wv = wc/2+np.sqrt(wc**2/4+wd**2) # voigt hw approx of Whiting 1968
        width = 40 * wv # 40 of the approximate half-widths
        # hard-code 1000 points
        xarr = np.linspace(freq-width, freq+width, 1.e3)

    lineshape = oh.voigt(xarr, 1., freq, sigma, gamma, True)

    # (2) calculate pop density
    # use HITRAN values for J, E_low; determine Qrot with parameterization
    # given in Dorn et al. of Goldman and Gillis (1981) data
    Qrot = (1.42e-6)*T**2 + 0.1485*T - 4.1
    popdens = (2*Ja+1)/Qrot*np.exp(-E_low/(k*T))

    # (3) total integrated absorption cross-section, cm^2
    # use Eq 3 in Dorn et al., using c in cm for result in cm^2
    sigma_tot = (1/(8*pi*c*100*wnum**2)*(2*Jb+1)/(2*Ja+1)*Aba) # cm^2
    # or Table 2.2 in Demtroeder p 41:
    # sigma_ij = (gj/gi)*c**2/(8*freq**2*d_freq) * Aji
    # sigma_tot = integrate sigma_ij d_freq
    # sigma_tot = (gj/gi)*c**2/(8*freq**2) * Aji
    # equivalent, with conversion wnum = freq / c

    # effective cross-section depends on population in lower state,
    # 'total' integrated cross-section, and lineshape (Voigt)
    sigma_eff = popdens * sigma_tot * lineshape # Dorn et al, Eq 7
    lineseries = pd.Series(sigma_eff, index=xarr)
    # cut off first and last entry to zero to avoid interpolation error later
    lineseries.iloc[0] = 0
    lineseries.iloc[-1] = 0
    lineseries.index.name = "Frequency, Hz"
    return lineseries

def simspec(hitlines, press=oh.OP_PRESS, T=oh.TEMP):
    '''Combine set of hitlines into spectrum, as pandas DataFrame.

    Call `simline` for each entry in input, without specifying frequency range
    over which to calculate absorption feature. Bundles up each pair of
    frequency values and absorption cross sections returned by `simline` into
    one pandas DataFrame object. Interpolate up to 5 consecutive NaN values in
    each line, which can arise from overlapping lines. Then return the
    DataFrame.

    The DataFrame index is a combination of all frequency values used for all
    the lines, but the DataFrame is very sparse, saving on memory.

    Parameters
    ----------
    hitlines : ndarray
    recarray in the format that loadhitran.processhitran spits out

    Outputs
    -------
    xarr : ndarray
    1D array of frequency values the spectrum was calculated over, Hz
    sigma_eff_array : ndarray
    2D array of effective absorption cross-section values across xarr for each
    line in hitlines, cm^2
    '''
    linedict = {}
    for line in hitlines:
        linedict.update({line.label: simline(line, None, press, T)})
    specdata = pd.DataFrame(linedict)
    specdata.interpolate(method='linear', limit=5, inplace=True)
    return specdata

def makeindexnm(specdata):
    '''Given spectrum with frequency index, make with wavelength (nm) index.
    '''
    nmindex = c/specdata.index.values*1e9
    specdata_nm = pd.DataFrame(data=specdata.values, index=nmindex,
                               columns=specdata.columns)
    specdata_nm.sort_index(inplace=True)
    return specdata_nm

def spectocsv(csvfile, specdata):
    '''write given specdata DataFrame to a csv file

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

    Outputs
    -------
    specdata : pd.DataFrame
    DataFrame containing spectrum
    '''
    specdata = pd.read_csv(csvfile, index_col=0)
    return specdata
