# -*- coding: utf-8 -*-
"""
Created on Tue May 27 16:10:34 2014

@author: abirdsall
"""
from . import atmcalcs as atm
from . import ohcalcs as oh

import numpy as np
import logging
from fractions import Fraction

LOADHITRAN_LOGGER = logging.getLogger('popmodel.loadhitran')

def importhitran(hfile, columns=None):
    '''
    Extract complete set of data from HITRAN-type par file.

    All HITRAN molecules have the same fixed-character fields here.

    Because each column has a different data type, the resulting array is 1D,
    with each entry consisting of all the entries for a specific feature.

    PARAMETERS:
    -----------
    hfile : str
    Input HITRAN file (160-char format)
    columns : tuple
    Column numbers to keep (default: all), e.g., (2, 3, 4, 7).

    OUTPUTS:
    --------
    hdata : ndarray
    Raw 1D ndarray with labels for each data entry. See HITRAN/JavaHawks
    documentation for explanation of each column.
    '''
    hdata = np.genfromtxt(hfile,
                         delimiter=(2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15,
                                    15, 15, 6, 12, 1, 7, 7),
                         dtype=[('molec_id', '<i4'),
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
                         usecols=columns)
    return hdata

def filterhitran(hfile, scutoff=1e-20, vabmin=3250, vabmax=3800,
                 columns=(0, 2, 3, 4, 5, 7, 10, 11, 12, 13, 17, 18)):
    '''
    Filter lines from HITRAN-type par file by intensity and wavenumber.

    Only return subset of fields for each line that are needed elsewhere.

    PARAMETERS:
    -----------
    hfile : str
    Input HITRAN file (160-char format).

    scutoff : float
    Minimum absorbance intensity cutoff, HITRAN units.

    vabmin : float
    Low end of desired wavenumber range, cm^-1.

    vabmax : float
    High end of desired wavenumber range, cm^-1.

    columns : tuple
    Column numbers to keep (default: subset I've found convenient).

    OUTPUTS:
    --------
    data_filter : ndarray
    Labeled array containing columns molec_id, wnum_ab, S, A, g_air, E_low,
    ugq, lgq, ulq, llq, g_up, g_low.
    '''
    data = importhitran(hfile, columns)

    wavnuminrange = np.logical_and(data['wnum_ab'] >= vabmin,
                                   data['wnum_ab'] <= vabmax)
    data_filter = data[np.logical_and(data['S'] >= scutoff, wavnuminrange)]
    return data_filter

def extractnjlabel_h2o(hdata):
    '''
    Extract J quantum number info and unique label from HITRAN for H2O.

    For global quanta, H2O is "class 6": non-linear triatomic, with three
    vibrational modes Global quanta have final 6 characters for quanta in the
    three vibrational modes.

    For local quanta, H2O is "group 1": asymmetric rotors. Three characters for
    J (total angular momentum, without nuclear spin), three for Ka, three for
    Kc, five for F (total angular momentum, including nuclear spin), one for
    Sym.

    Label is in form "[Ja]_[Jb]_[wnum]". Including wnum ensures unique labels.

    PARAMETERS:
    -----------
    hdata : ndarray
    Must contain HITRAN (140-char format) information about quantum states, as
    processed by importhitran.

    OUTPUTS:
    --------
    Ja, Jb, label : ndarrays (3)
    J quantum numbers for 'a' and 'b' states, and strings identifying the b <--
    a transitions.
    '''
    llq = hdata['llq']
    ulq = hdata['ulq']

    Ja = np.asarray([float(entry[:3]) for entry in llq])
    Jb = np.asarray([float(entry[3:6]) for entry in ulq])

    label = np.vectorize(lambda x, y, z: x+'_'+y+'_'+z) \
            (Ja.astype('int').astype('str'), Jb.astype('int').astype('str'),
             hdata['wnum_ab'].astype('str'))
    return Ja, Jb, label

def extractnjlabel(hdata):
    '''
    Extract N and J quantum number and unique label from HITRAN for OH.

    Determine Na from the spin and J values provided in HITRAN, where
    J = N + spin (spin = +/-1/2). Determine Nb from Na and the P/Q/R branch.

    PARAMETERS:
    -----------
    hdata : ndarray
    Must contain HITRAN (140-char format) information about quantum states, as
    processed by importhitran.

    OUTPUTS:
    --------
    Na, Nb, Ja, Jb, label : ndarrays (5)
    N and J quantum numbers for 'a', and 'b' states, and strings identifying
    the b <-- a transitions. Format of label is 'X_#(*)ll' where X denotes
    branch (P, Q, R, ...), # describes J cases of upper and lower states (1, 2,
    12, 21), * is lower state N, and ll describes which half of lambda doublet
    is upper/lower state (ef, fe, ee, ff).
    '''
    # shorthand for HITRAN entries of interest, in hdata
    lgq = hdata['lgq']
    llq = hdata['llq']
    ugq = hdata['ugq']

    # extract spin values: 3/2 denotes spin + 1/2, 1/2 denotes spin -1/2
    spinsa = np.asarray([float(Fraction(entry[8:11])) for entry in lgq]) - 1
    spinsb = np.asarray([float(Fraction(entry[8:11])) for entry in ugq]) - 1
    # extract total angular momentum Ja
    Ja = np.asarray([float(entry[4:8]) for entry in llq])
    Na = Ja - spinsa

    # extract splitting info
    efsplit = np.asarray([entry[8:10] for entry in llq])

    # extract b state N and J from provided branch values
    # OH has two Br values, for N and J, which differ only when spin states
    # change. Verified first value is N by seeing that 'QP' happens when lower
    # state is X3/2 and upper is X1/2. This is only consistent with Q referring
    # to N and P referring to J, not vice versa.
    br_dict = {'O':-2, 'P':-1, 'Q':0, 'R':1, 'S':2}
    br_N = np.vectorize(lambda y: y[1])(llq)
    br_J = np.vectorize(lambda y: y[2])(llq)
    br_N_value = np.asarray([br_dict[entry] for entry in br_N])
    br_J_value = np.asarray([br_dict[entry] for entry in br_J])

    # Transition name. Uses nomenclature of Dieke and Crosswhite, J Quant
    # Spectrosc Radiat Transfer 2, 97-199, 1961.
    # index differentiates between two components of doublet from electronic
    # spin considerations: '1' means J = N+1/2, '2' means J = N-1/2, '12' means
    # transition is between '1' (upper) & '2' (lower), and vice versa for '21'
    index_dict = {'1': (spinsa == spinsb) & (spinsa == 0.5),
                  '2': (spinsa == spinsb) & (spinsa == -0.5),
                  '21': (spinsa != spinsb) & (spinsa == 0.5),
                  '12': (spinsa != spinsb) & (spinsa == -0.5)}
    indexarray = np.empty_like(spinsa, dtype='str')
    for label, entry in index_dict.iteritems():
        indexarray[np.where(entry)] = label
    # bring it all together into a single 'label' string per line
    label = np.vectorize(lambda x, y, z, w: x+'_'+y+'('+z+')'+w) \
            (br_N, indexarray, Na.astype('int').astype('str'), efsplit)

    Nb = Na + br_N_value    # N quantum number for b state
    Jb = Ja + br_J_value    # J quantum number for b state

    return Na, Nb, Ja, Jb, label

def processhitran(hfile, scutoff=1e-20, vabmin=3250, vabmax=3800):
    '''
    Extract parameters needed for IR-UV LIF kinetics modeling from HITRAN file.

    Extract N quantum numbers, UV energies, Einstein coefficients, Doppler
    broadening, quenching rate constants, beam parameters.

    Use functions and parameters in 'atmcalcs' and 'ohcalcs' modules.

    Parameters:
    -----------
    hfile : str
    Input HITRAN file (160-char format).

    scutoff : float
    Minimum absorbance intensity cutoff, HITRAN units.

    vabmin : float
    Low end of desired wavenumber range, cm^-1.

    vabmax : float
    High end of desired wavenumber range, cm^-1.

    Outputs:
    --------
    alldata : ndarray
    Labeled array containing columns wnum_ab, S, g_air, E_low, ga, gb, Aba,
    Bba, Bab, fwhm_dop, Na, Nb, Ja, Jb, label
    '''
    # Extract parameters from HITRAN
    hdata = filterhitran(hfile, scutoff, vabmin, vabmax)

    vab = atm.WAVENUM_TO_HZ*hdata['wnum_ab']

    # Extract and calculate Einstein coefficients. See ohcalcs.py for details
    # on convention used for calculating B coefficients.
    Aba = hdata['A']
    ga = hdata['g_low']
    gb = hdata['g_up']

    Bba = oh.b21(Aba, vab)
    Bab = oh.b12(Aba, ga, gb, vab)

    if hdata['molec_id'][0] == 13: # OH
        Na, Nb, Ja, Jb, label = extractnjlabel(hdata)
        fwhm_dop = oh.fwhm_doppler(vab, oh.TEMP, oh.MASS)

    elif hdata['molec_id'][0] == 1:  # H2O
        Ja, Jb, label = extractnjlabel_h2o(hdata)
        # just make everything else -1s
        Na, Nb, fwhm_dop = [np.ones_like(hdata['A'])*(-1)]*3

    else:
        raise ValueError("Unsupported molecule type ", hdata['molec_id'][0])

    arraylist = [hdata['wnum_ab'],
                 hdata['S'],
                 hdata['g_air'],
                 hdata['E_low'],
                 hdata['g_low'],
                 hdata['g_up'],
                 Aba,
                 Bba,
                 Bab,
                 fwhm_dop,
                 Na,
                 Nb,
                 Ja,
                 Jb,
                 label]

    dtypelist = [('wnum_ab', 'float'),
                 ('S', 'float'),
                 ('g_air', 'float'),
                 ('E_low', 'float'),
                 ('ga', 'int'),
                 ('gb', 'int'),
                 ('Aba', 'float'),
                 ('Bba', 'float'),
                 ('Bab', 'float'),
                 ('fwhm_dop', 'float'),
                 ('Na', 'int'),
                 ('Nb', 'int'),
                 ('Ja', 'int'),
                 ('Jb', 'int'),
                 ('label', label.dtype)]

    alldata = np.rec.fromarrays(arraylist, dtype=dtypelist)
    LOADHITRAN_LOGGER.info('processhitran: file processed')
    return alldata
