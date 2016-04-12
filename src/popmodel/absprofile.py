'''popmodel module for AbsProfile class
'''
from __future__ import division
from . import ohcalcs as oh

import logging
import numpy as np
from scipy.constants import k as kb
from scipy.constants import c

class AbsProfile(object):
    '''absorbance line profile, initially defined in __init__ by a center
    wavenumber `wnum` and a `binwidth`. Calling self.makeprofile then generates
    two 1D arrays:

    abs_freq : bins of frequencies (Hz)
    pop : relative population absorbing in each frequency bin

    pop is generated from abs_freq and the Voigt profile maker ohcalcs.voigt,
    which requires parameters that are passed through as makeprofile arguments
    (default are static parameters in ohcalcs). The formation of the two arrays
    is iterative, widening the abs_freq range by 50% until the edges of the pop
    array have less than 1% of the center.
    '''
    def __init__(self, wnum, binwidth=1.e6):
        self.logger = logging.getLogger('popmodel.absprofile.AbsProfile')
        self.wnum = wnum # cm^-1
        self.freq = wnum*c*100 # Hz
        self.binwidth = binwidth # Hz
        # instance attributes calculated in makeprofile
        self.abs_freq = None
        self.pop = None
        self.fwhm = None
        # intpop calculated in Sweep.alignbins() call
        self.intpop = None

    def __str__(self):
        return 'Absorbance feature centered at '+str(self.wnum)+' cm^-1'

    def makeprofile(self, abswidth=1000.e6, press=oh.OP_PRESS, T=oh.TEMP,
                    g_air=oh.G_AIR, mass=oh.MASS, edgecutoff=0.01):
        ''' Use oh.voigt to create absorption profile.

        Writes to self.abs_freq and self.pop.

        Parameters:
        -----------
        abswidth : float
        Minimum width of profile, Hz. Starting value that then expands if this
        does not capture 'enough' of the profile (defined as <1% of peak height
        at edges).

        press : float
        Operating pressure, torr. Defaults to ohcalcs value

        T : float
        Temperature. Defaults to ohcalcs value

        g_air : float
        Air-broadening coefficient provided in HITRAN files, cm^-1 atm^-1.
        Defaults to ohcalcs value.

        mass : float
        Mass of molecule of interest, kg. Defaults to ohcalcs value

        edgecutoff : float
        Cut-off for acceptable relative intensity at edge of profile compared
        to peak. If relative intensity is larger than this value, Voigt profile
        will be recalculated over a 50% broader frequency range.
        '''
        sigma = (kb*T / (mass*c**2))**(0.5)*self.freq # Gaussian std dev

        gamma = (g_air*c*100) * press/760. # Lorentzian parameter
        # air-broadened HWHM at 296K, HITRAN (converted from cm^-1 atm^-1)
        # More correctly, correct for temperature -- see Dorn et al. Eq 17

        # Make abs_freq profile, checking pop at edge <1% of peak
        enoughwidth = False
        while enoughwidth == False:
            abs_freq = np.arange(-abswidth/2,
                                 abswidth/2+self.binwidth,
                                 self.binwidth)
            raw_pop = oh.voigt(abs_freq, 1, 0, sigma, gamma, True)
            norm_factor = 1/np.sum(raw_pop)
            pop = raw_pop * norm_factor # makes sum of pops = 1.
            if pop[0] >= edgecutoff*np.max(pop):
                abswidth = abswidth*1.5
            else:
                enoughwidth = True
        self.abs_freq = abs_freq
        self.pop = pop
        startfwhm = abs_freq[pop >= np.max(pop)*0.5][0]
        endfwhm = abs_freq[pop >= np.max(pop)*0.5][-1]
        self.fwhm = endfwhm - startfwhm
        self.logger.info('makeprofile: made abs profile')
        self.logger.info('makeprofile: abs profile has FWHM = %.2g MHz',
                         self.fwhm/1e6)
        self.logger.info('makeprofile: total width of stored array = %.2g MHz',
                         abswidth/1e6)
