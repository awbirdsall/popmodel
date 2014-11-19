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

# Following deprecated -- as of 14 Aug 2014, see 'popmodel'

# # ODE parameters:
# delay = 150e-9    # s, artificial UV delay in model for a and b to equilibrate.
# tl = 300e-9 # total integration time
# dt = 1e-9
# t_steps = int(round(tl/dt)) # does not include t=0 entry
# t = np.linspace(0, tl, t_steps+1)
# y0 = [1.,0,0]

# def solveode(file, a):
#     '''Integrate ode describing IR-UV two-photon LIF, given master equation
#     (with its Jacobian) and all relevant parameters. Use 'ohcalcs' and
#     'atmcalcs' modules.
    
#     Define global parameters that are independent of HITRAN OH IR data within
#     function: Additional OH parameters related to 'c' state and quenching, and
#     laser parameters. Also set up parameters for solving and plotting ODE.

#     Parameters:
#     -----------
#     file : str
#     Input HITRAN file (160-char format).
    
#     a : int
#     index of transition to use, within 'file'.
    
#     Outputs:
#     --------
#     N : ndarray
#     Relative population of 'a', 'b' and 'c' states over integration time.    
#     '''
#     # Collect info from HITRAN and extract:
#     x = processHITRAN(file)
#     Bba = x['Bba']
#     Bab = x['Bab']
#     Bcb = x['Bcb']
#     Bbc = x['Bbc']

#     params = (oh.Aca, oh.Acb, Bab[a], Bba[a], Bbc[a], Bcb[a], oh.kqb, oh.kqc,
#               oh.Q, oh.Lab, oh.Lbc, oh.period_UV, oh.pulsewidth_UV, delay)

#     r = ode(dN, jac)
#     r.set_integrator('lsoda', with_jacobian=True)
#     r.set_initial_value(y0, 0)
#     r.set_f_params(*params)
#     r.set_jac_params(*params)
#     N = y0
#     # Solve ODE
#     while r.successful() and r.t < tl:
#         nextstep = r.integrate(r.t + dt)
#         N = np.concatenate((N, nextstep)) 
#     N = np.resize(N, [np.size(N)/3,3])
#     return N

#     # Alternate 'odeint' approach -- y and t need to be swapped in f, Dfun:    
#     #    y0 = [1,0,0]
#     #    t_step = np.linspace(0, 1e-7, 1000)
#     #    fnc_args = (Aca, Acb, Bab[0], Bba[0], Bbc[0], Bcb[0], kqb, kqc, Q,
#     #                 Wab[0], sp_rad_bc[0], period_UV, pulsewidth_UV,)
#     #    r, infodict = odeint(dN, y0, t_step, args=fnc_args, Dfun = jac, full_output=True) 
#     #    return r, infodict

# def plotode(N):
#     # Plot ODE solution
#     ground, vib, fluor = N.T

#     fig = plt.figure()
#     p1 = fig.add_subplot(111)
#     p1.plot(t, N)
#     plt.axvline(x=delay, ls='--')
#     plt.axvline(x=delay+oh.pulsewidth_UV, ls='--')
#     p1.set_xlim(0,(delay+oh.pulsewidth_UV)*1.1)
#     plt.legend(('ground', 'vib', 'fluor', 'turn on UV', 'turn off UV'))
#     plt.grid()

#     # inset axes of the 'c' state
#     inset = fig.add_axes([.5, .25, .3, .3])
#     inset.plot(t,fluor)
#     plt.title('Fluorescing state')
#     plt.locator_params(axis='y', nbins=3)
#     inset.set_xlabel('Time')
#     inset.set_ylim(0,np.amax(fluor)*1.1)
#     plt.show()
#     # To save:
#     # figfile = raw_input('save image as .png: ') # use raw_input with Python 2
#     # if figfile[-4:]=='.png':
#     #     filename = figfile
#     # else:
#     #     filename = figfile+'.png'
#     # plt.savefig(filename, bbox_inches='tight', transparent=True, dpi=72)
    
#     # N.B.: population 'inversion' b/w 'a' and 'b' possible due to difference
#     # in degeneracies. Will only see this when quenching very small.

# def fluoryield(N,x):
#     '''
#     Calculate fluorescence yield given transition's quantum yield and 
#     population kinetics, using maximum population in fluorescing state.
#     '''
#     qyield = x['qyield']
#     fyield = qyield*N[(delay+oh.pulsewidth_UV)/tl*t_steps,2]
#     return fyield

# def dN(t, y, Aca, Acb, Bab, Bba, Bbc, Bcb, kqb, kqc, Q, Lab, Lbc, 
#        period, pulsewidth_UV, delay):
#     # Pulse the UV (b to c) laser (assume total modeled time < rep rate):
#     if t>delay and t<pulsewidth_UV+delay:
#         Lbc=Lbc
#     else:
#         Lbc = 0

#     # Coefficients related to
#     # b<--a processes:
#     absorb_ab = Bab * Lab * y[0]

#     # b-->a processes:
#     stim_emit_ba = Bba * Lab * y[1]
#     quench_b = kqb * Q * y[1]

#     # c<--b processes:
#     absorb_bc = Bbc * Lbc * y[1]

#     # c-->a processes:
#     spont_emit_ca = Aca * y[2]
#     quench_c = kqc * Q * y[2]

#     # c-->b processes:
#     stim_emit_cb = Bcb * Lbc * y[2]
#     spont_emit_cb = Acb * y[2]

#     N0 = - absorb_ab + stim_emit_ba + spont_emit_ca + quench_b + quench_c
#     N1 = absorb_ab - stim_emit_ba + stim_emit_cb - absorb_bc - quench_b + spont_emit_cb
#     N2 = - spont_emit_cb + absorb_bc - spont_emit_ca - quench_c - stim_emit_cb
#     return np.array([N0, N1, N2])

# def jac(t, y, Aca, Acb, Bab, Bba, Bbc, Bcb, kqb, kqc, Q, Lab, Lbc, 
#         period, pulsewidth_UV, delay):
#     '''Jacobian matrix for dN. Across first row is dN0/dy[i].'''
    
#     # Pulse the UV (b to c) laser (assume total modeled time < rep rate):
#     if t>delay and t<pulsewidth_UV+delay:
#         Lbc=Lbc
#     else:
#         Lbc = 0

#     return np.array([[-Bab*Lab, Bba*Lab+kqb*Q, Aca+kqc*Q],
#                      [Bab*Lab, -Bba*Lab-Bbc*Lbc-kqb*Q, Bcb*Lbc+Acb],
#                      [0, Bbc*Lbc, -Acb-Bcb*Lbc-Aca-kqc*Q]])
