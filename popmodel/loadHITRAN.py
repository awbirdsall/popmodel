# -*- coding: utf-8 -*-
"""
Created on Tue May 27 16:10:34 2014

@author: abirdsall

Next steps:
Degeneracy of c state?
"""
import atmcalcs as atm

import numpy as np
import ohcalcs as oh
import logging
from fractions import Fraction

def importhitran(file, columns=None):
    '''
    Extract complete set of data from HITRAN-type data file. Because each
    column has a different data type, the resulting array is 1D, with each
    entry consisting of all the entries for a specific feature.
    
    PARAMETERS:
    -----------
    file : str
    Input HITRAN file (160-char format)
    columns : tuple
    Column numbers to keep (default: all), e.g., (2, 3, 4, 7).
    
    OUTPUTS:
    --------
    data : ndarray
    Raw 1D ndarray with labels for each data entry. See HITRAN/JavaHawks
    documentation for explanation of each column.
    '''
    data = np.genfromtxt(file,
                        delimiter = (2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15,
                                     15, 15, 6, 12, 1, 7, 7),
                        dtype=[('molec id', '<i4'),
                        ('isotop', '<i4'),
                        ('wnum_ab', '<f8'),
                        ('S', '<f8'), 
                        ('A', '<f8'), 
                        ('g_air', '<f8'),
                        ('g_self', '<f8'),
                        ('E_low', '<f8'),
                        ('n_air', '<f8'),
                        ('delta_air', '<f8'),
                        ('ugq', 'S15'),
                        ('lgq', 'S15'),
                        ('ulq', 'S15'),
                        ('llq', 'S15'),
                        ('ierr', 'S6'),
                        ('iref', 'S12'),
                        ('flag', 'S1'),
                        ('g_up', 'f8'),
                        ('g_low', 'f8')],
                        usecols = columns)
                        
    return data

def filterhitran(file, Scutoff=1e-20, vabmin=3250, vabmax=3800):
    '''
    Extract select subset of data from HITRAN-type data file, based on cutoff
    intensity and range of wave numbers.
    
    PARAMETERS:
    -----------
    file : str
    Input HITRAN file (160-char format).

    Scutoff : float
    Minimum absorbance intensity cutoff, HITRAN units.

    vabmin : float
    Low end of desired wavenumber range, cm^-1.

    vabmax : float
    High end of desired wavenumber range, cm^-1.
    
    OUTPUTS:
    --------
    data_filter : ndarray
    Labeled array containing columns wnum_ab, S, A, g_air, E_low, ugq, lgq,
    ulq, llq, g_up, g_low.
    '''
    data = importhitran(file, (2, 3, 4, 5, 7, 10, 11, 12, 13, 17, 18))

    wavnuminrange = np.logical_and(data['wnum_ab']>=vabmin,
            data['wnum_ab']<=vabmax)
    data_filter = data[np.logical_and(data['S']>=Scutoff, wavnuminrange)]
    return data_filter

def extractN(x):
    '''
    Extract N quantum number info from HITRAN quantum state data for OH.

    Determine Na from the spin and J values provided in HITRAN, where
    J = N + spin (spin = +/-1/2). Determine Nb from Na and the P/Q/R branch.

    Determine Nc assuming the P branch transition will be used for c<--b.

    PARAMETERS:
    -----------
    x : ndarray
    Must contain HITRAN (140-char format) information about quantum states, as
    processed by importhitran.

    OUTPUTS:
    --------
    Na : ndarray
    N quantum numbers for 'a' state.

    Nb : ndarray
    N quantum numbers for 'b' state.

    Nc : ndarray
    N quantum numbers for 'c' state.
    '''
    lgq = x['lgq']
    llq = x['llq']
    spins = np.asarray([float(Fraction(entry[8:11])) for entry in lgq]) - 1
    # 3/2 is spin + 1/2, 1/2 is spin -1/2
    J = np.asarray([float(entry[4:8]) for entry in llq])
    Na = J - spins

    br = {'O':-2, 'P':-1, 'Q':0, 'R':1, 'S':2}
    branches = np.asarray([br[entry[2]] for entry in llq])
    # OH has two Br values, for N and J, which differ only when spin states
    # change. Here, looking at the second one (N?)

    Nb = Na + branches    # N quantum # for b state

    Nc = Nb - 1    # Assuming P branch transition is most efficient for c<--b.
    return Na, Nb, Nc

def calculateUV(Nc, wnum_ab, E_low):
    '''
    Calculate c<--b transition wavenumber, accounting for rotational states.
    Fails if Nc>4, so need to filter out high N transitions from HITRAN first
    -- intensity cutoff should be fine.

    PARAMETERS:
    -----------
    Nc : ndarray
    N quantum numbers for 'c' state.

    wnum_ab : ndarray
    Wavenumbers of b<--a transition, cm^-1.

    E_low : ndarray
    Energy level of 'a' state, cm^-1

    OUTPUTS:
    --------
    wnum_bc : ndarray
    Wavenumbers of c<--b transition, cm^-1.
    '''
    # dict of N'-dependent c-state energy, cm^-1
    # Using Erin/Glenn's values from 'McGee' for v'=0 c-state
    E_cdict = {4:32778.1, 3:32623.4, 2:32542, 1:32474.5, 0:32778.1}    
    # use dict to choose appropriate E_c
    E_c = np.asarray([E_cdict[entry] for entry in Nc])    # Error if Nc>4 ...
    wnum_bc = E_c - wnum_ab - E_low
    return wnum_bc

def processHITRAN(file):
    '''
    Extract parameters needed for IR-UV LIF kinetics modeling from HITRAN
    file: N quantum numbers, UV energies, Einstein coefficients, Doppler
    broadening, quenching rate constants, beam parameters.
    
    Use functions and parameters in 'atmcalcs' and 'ohcalcs' modules.

    Parameters:
    -----------
    file : str
    Input HITRAN file (160-char format).
    
    Outputs:
    --------
    alldata : ndarray
    Labeled array containing columns wnum_ab, wnum_bc, S, A, g_air, E_low, ga,
    gb, Aba, Bba, Bab, FWHM_Dop_ab, FWHM_Dop_bc
    '''
    # Extract parameters from HITRAN
    x = filterhitran(file)

    Na, Nb, Nc = extractN(x)

    wnum_bc = calculateUV(Nc, x['wnum_ab'], x['E_low'])
    
    # Perform calculations using transition frequencies, Hz.
    vbc = atm.wavenum_to_Hz*wnum_bc
    vab = atm.wavenum_to_Hz*x['wnum_ab']

    # Extract and calculate Einstein coefficients. See ohcalcs.py for details
    # on convention used for calculating B coefficients.
    Aba = x['A']
    ga = x['g_low']
    gb = x['g_up']

    Bba = oh.b21(Aba, vab)
    Bab = oh.b12(Aba, ga, gb, vab)
    
    # Remaining Einstein coefficients:
    # Assuming same Acb regardless of b and c rotational level. Could do better
    # looking at a dictionary of A values from HITRAN. Not a high priority to
    # improve since not currently using UV calcs. TODO
    Bcb = oh.b21(oh.Acb, vbc)
    Bbc = oh.b12(oh.Acb, gb, oh.gc, vbc)

    # Collision broadening:
    FWHM_Dop_ab = oh.fwhm_doppler(vab, oh.temp, oh.mass)
    FWHM_Dop_bc = oh.fwhm_doppler(vbc, oh.temp, oh.mass)

    # Quantum yield:
    qyield = oh.Aca/(oh.Aca + Bcb*oh.Lbc + oh.Q*oh.kqc)

    # Transition name:
    branch = np.vectorize(lambda y: y[1:2])(x['llq'])
    line = np.vectorize(lambda y: y[5:8])(x['llq'])

    arraylist = [x['wnum_ab'],
                wnum_bc,
                x['S'],
                x['g_air'],
                x['E_low'],
                x['g_low'],
                x['g_up'],
                Aba,
                Bba,
                Bab,
                Bcb,
                Bbc,
                FWHM_Dop_ab,
                FWHM_Dop_bc,
                qyield,
                branch,
                line,
                Na,
                Nb,
                Nc]

    dtypelist = [('wnum_ab','float'),
                ('wnum_bc','float'),
                ('S','float'),
                ('g_air', 'float'),
                ('E_low','float'),
                ('ga','int'),
                ('gb','int'),
                ('Aba', 'float'),
                ('Bba', 'float'),
                ('Bab', 'float'),
                ('Bcb', 'float'),
                ('Bbc', 'float'),
                ('FWHM_Dop_ab', 'float'),
                ('FWHM_Dop_bc', 'float'),
                ('qyield', 'float'),
                ('branch', '|S1'),
                ('line', '|S3'),
                ('Na','int'),
                ('Nb','int'),
                ('Nc','int')]

    alldata = np.rec.fromarrays(arraylist,dtype=dtypelist)
    logging.info('processHITRAN: file processed')
    return alldata
