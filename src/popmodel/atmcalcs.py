## Python module for atmospheric chemistry calculations
## Updated May 2014
## Adam Birdsall

from scipy.constants import c, N_A, gas_constant, atm, torr

## Constants derived from scipy.constants:

r_gas = gas_constant/(atm/1000.)    # ideal gas constant, L atm K^-1 mol ^-1

wavenum_to_Hz = c*100   # one wavenumber in Hz

torr_to_atm = torr/atm    # one torr in atm

## Conversions

def wavenum_to_nm(wavenum):
    '''Converts input from wavenumber to nm'''
    nm = c / (wavenum * wavenum_to_Hz) * 10**9
    return nm

def nm_to_wavenum(nm):
    '''Converts input from nm to wavenumber'''
    wavenum = c / (nm * wavenum_to_Hz) * 10**9
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
    n_air = N_A * press * torr_to_atm / (r_gas * 1000 * temp)
    numdens = n_air * mix
    return numdens

def press_to_numdens(press=760, temp=273):
    '''Convert pressure (torr) and temp (K) to num density (molec cm^-3)
    '''
    numdens = (press * torr_to_atm) / (r_gas * temp) * (N_A / 1000)
    return numdens


## Executes when running from within module

if __name__ == "__main__":
    print(mix_to_numdens(.5e-12, 2.5, 296))
