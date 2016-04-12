'''popmodel module for 'atmospheric chemistry'-type calculations
'''
from __future__ import division
from scipy.constants import c, N_A, gas_constant, atm, torr

## Constants derived from scipy.constants:

R_GAS = gas_constant/(atm/1000.)    # ideal gas constant, L atm K^-1 mol ^-1

WAVENUM_TO_HZ = c*100   # one wavenumber in Hz

TORR_TO_ATM = torr/atm    # one torr in atm

## Conversions

def wavenum_to_nm(wavenum):
    '''Converts input from wavenumber to nm'''
    wlnm = c / (wavenum * WAVENUM_TO_HZ) * 10**9
    return wlnm

def nm_to_wavenum(wlnm):
    '''Converts input from nm to wavenumber'''
    wavenum = c / (wlnm * WAVENUM_TO_HZ) * 10**9
    return wavenum

def mix_to_numdens(mix, press=760, temp=273):
    '''Converts input from mixing ratio to number density.

	Parameters
	----------
	mix : float
	Mixing ratio.
	press : float
	Pressure in torr, default 760.
	temp : float
	Temperature in K, default 298

	Returns
	-------
	numdens : float
	Number density in molecules cm^-3
	'''
    n_air = N_A * press * TORR_TO_ATM / (R_GAS * 1000 * temp)
    numdens = n_air * mix
    return numdens

def press_to_numdens(press=760, temp=273):
    '''Convert pressure (torr) and temp (K) to num density (molec cm^-3)
    '''
    numdens = (press * TORR_TO_ATM) / (R_GAS * temp) * (N_A / 1000)
    return numdens
