## Python module for atmospheric chemistry calculations
## Updated May 2014
## Adam Birdsall


## Physical constants

avog = 6.0221e23    # Avogadro's number

r_gas = 0.08205746    # ideal gas constant, L atm K^-1 mol ^-1

c_light = 2.9979e8    # speed of light, m s^-1

h = 6.6260755e-34    # Planck's constant, J s

wavenum_to_Hz = 2.9979e10    # one wavenumber in Hz

wavenum_to_MHz = 2.9979e4    # one wavenumber in MHz

torr_to_atm = 1/760.    # one torr in atm

## Conversions

def wavenum_to_nm(wavenum):
	'''Converts input from wavenumber to nm'''
	nm = c_light / (wavenum * wavenum_to_Hz) * 10**9
	return nm

def nm_to_wavenum(nm):
	'''Converts input from nm to wavenumber'''
	wavenum = c_light / (nm * wavenum_to_Hz) * 10**9
	return wavenum

def mix_to_numdens(mix, press=760, temp=273):
	'''
	Converts input from mixing ratio to number density.

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
	n_air = avog * press * torr_to_atm / (r_gas * 1000 * temp)
	numdens = n_air * mix
	return numdens

def press_to_numdens(press=760, temp=273):
    '''input pressure in torr and temp in K; output num density in molecules cm^-3'''
    numdens = (press * torr_to_atm) / (r_gas * temp) * (avog / 1000)
    return numdens


## Executes when running from within module

if __name__ == "__main__":
    print(mix_to_numdens(.5e-12, 2.5, 296))