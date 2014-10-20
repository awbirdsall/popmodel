## Python module for OH double resonance calculations
## Updated June 2014
## Adam Birdsall

## Literature cited:

# Bradshaw et al, Appl Opt, 23, 2134 (1984)
# Daily et al., Applied Optics 16 (3), 568-571 (1977)
# Fuchs et al., Atmos Meas Tech, 1611-1626(2012)
# Schlosser et al., J Atmos Chem, 56, 187-205 (2007)
# Tsuji et al, Bull Chem Soc Jpn, 73, 2695-2702 (2000)
# van de Meerakker et al, Phys Rev Lett, 95, 013003 (2005)


import math as m
import atmcalcs as atm
import numpy as np
from scipy.constants import k as kb
from scipy.constants import c, N_A, pi

#######################

def kqavg(kn2,ko2,kh2o,xh2o=0.02):
    '''get average k in air (default 0.78 N2/0.20 O2/0.02 H2O)'''
    xn2=0.79-xh2o/2
    xo2=0.21-xh2o/2
    return xn2*kn2+xo2*ko2+xh2o*kh2o

#######################

## Literature values for OH
mass = 17.01/(N_A)*1000    # kg

nu12 = 3407.53 * atm.wavenum_to_Hz  # Hz, IR transition used in Tsuji et al, 2000

# Degeneracies
# Generally, degeneracies for rotational quantum number J are 2J+1. The ground
# electronic state is PI, so lambda doubling doubles the degeneracies (unless
# it breaks them?). A-state is SIGMA, so there is no lambda doubling.
ga = 20    # (J = 4.5)

gb = 16    # (J = 3.5)
gc = 8    # From (2*J + 1) and N = 3 (?) for Tsuji et al. feature. Assuming
            # Q branch.

# Einstein coefficients
Aba = 16.9      # Einstein coefficient for spontaneous emission, s^-1;
                # using van de Meerakker et al, 2005
                # alt value: 14.176 s^-1 from Tsuji et al, 2000

Acb = 5300 #for A2Sigma+(v'=0)-->X2Pi(v"=1) Copeland (1987). Really, we'd 
# want the A's for each rotational transition...

Aca = 1.45e6 #s-1, for A2Sigma+(v'=0)-->X2Pi(v"=0), German (1975)
# No c<--a laser, so B coefficients not applicable.

# rotational relaxation
# assume for now dealing with N"=1.
rotfrac_a = 0.199104 # taken from LIFBASE, 296 K thermal distribution, both halves of lambda doublet
rotfrac_b = 0.192688 # this row and previous: N(F1e+f) for J=1.5
rotfrac_c = 0.130170 # v'=1, N(F1) for J=1.5 + N(F2) for J=0.5
rotfrac = np.array([rotfrac_a,rotfrac_b,rotfrac_c])

## HITRAN data for 3407 cm^-1 IR transition
## v = 0 --> v = 1, J = 4.5 --> J = 3.5

g_air = .053    # air-broadened HWHM, cm^-1 atm^-1 at 296K

g_self = .03    # self-broadened HWHM, cm^-1 atm^-1 at 296K

n_air = .66    # coeff of temp dependence of air-broadened half-width

# uncertainties in HITRAN data for this transition:
# in wavenumber: 3 ('0.001 - 0.01')
# in intensity: 4 ('10-20%'')
# in g-air: 2 ('avg or estimate')
# in g-self: 0 ('unreported')
# in n_air: 1 ('default')
# in air-pressure shift: 0 ('unreported') -- value given as 0

## Laser operating parameters

temp = 296.    # temperature, K. All HITRAN data assumes 296K!

beam_diam_IR = 5e-3    # beam diameter, m

beam_diam_UV = 5e-3    # beam diameter, m

bandwidth_IR = 1e6    # Hz, Aculight spec
bandwidth_UV = 7e9    # Hz, Schlosser et al 2007.

power_IR = 3    # W, Aculight spec
power_UV = 50e-3    # W, Schlosser et al 2007, average power

reprate_UV = 8.5e3    # Hz, Fuchs et al. 2012
period_UV = 1/reprate_UV    # s
pulsewidth_UV = 30e-9    #s, somewhat arbitrary, Fuchs et al. report 25 ns pulse
peak_power_UV = power_UV * period_UV/pulsewidth_UV # W, peak power

# Beam parameters
A_IR = pi*(.5*beam_diam_IR)**2 # beam area, m^2
A_UV = pi*(.5*beam_diam_UV)**2

# Spectral radiances (intensities) for master equations, W/m^2*sr*Hz
# Note IR is CW laser, while UV is pulsed.
Lab = power_IR/(A_IR*bandwidth_IR)
Lbc = peak_power_UV/(A_UV*bandwidth_UV)

op_press = 2.    # operating pressure of detection cell, torr
                # Julich LIF instrument operates at 
                # 3.5 hPa = 2.6 torr (Fuchs et al.)
                # 8.5 hPa = 6.4 torr (Schlosser et al.)
xoh = 0.5e-12 # 0.5 pptv, ballpark (1.2E7 cm^-3 at 760 torr, 296 K)

# Quenching:
xh2o = 0.02    # mole fraction of water
Q = atm.press_to_numdens(op_press, temp) # total 'quencher' conc, molec/cm^3

# Quenching speciated by N2/O2/H2O: (s^-1/(molec cm^-3))
# Full result of lit search in 'vib excited lifetime calcs OH TP LIF.xlsx'.
# Define estimated rate constants based on these values.
kqbh2o=1.36e-11    # Reported value for multiple groups, +/-0.4e-11; however, Silvente et al. give value 50% larger
kqbo2=1e-13    # Choose intermediate value between reported measurements of 7.5e-14 and 1.3e13
kqbn2=1.5e-15    # Only one reported measurement (D'Ottone et al.), but this term is a minor contribution
kqb = kqavg(kqbn2,kqbo2,kqbh2o,xh2o)

quenchb = kqb * Q # s^-1

# Electronic quenching of A state (s^-1/(molec cm^-3))
kqch2o = 68.0e-11 # Wysong et al. 1990, v'=0
kqco2 = 13.5e-11 # Wysong et al. 1990, v'=0
kqcn2 = 1.9e-11 # Copeland et al. 1985, v'=0, N'=3 (other N available -- gets smaller with bigger N)
kqc = kqavg(kqch2o,kqco2,kqch2o,xh2o)

quenchc = kqc * Q # s^-1

###############################

## Functions for calculations

# N.B. Two conventions for Einstein B coefficient:
# (1) Spectral radiance (intensity) L: J m^-2 s^-1 Hz^-1 sr^-1 = W m^-2 Hz^-1 sr^-1
#     units of B: m^2 J^-1 s^-1 = s kg^-1
# (2) Energy density rho: J m^-3 Hz^-1 sr^-1 = W s m^-3 Hz^-1 sr^-1
#     units of B: m^3 J^-1 s^-2 = m kg^-1
# We're using (1), intensity convention, e.g., Daily et al., 1977.
# L and rho are related by L = c*rho (see McQuarrie Ch. 15).
# L and rho are also per solid angle, or sr^-1 -- see Wikipedia.

def b21(a21, freq=nu12):
    '''
    Calculate Einstein coefficient for stimulated emission from A
    coefficient, using McQuarrie Eqn 15.13 and dividing by c to
    convert to intensity convention, m^2 J^-1 s^-1.

    Also consistent with Demtroder 3rd ed, assuming typo in eqn. 2.22.
    (i.e., missing nu^3 in numerator, h is to first power)
    Provides result consistent with Frank's value of 1e8 for OH
    excitation to v' = 1 from email 16 May 2014.

    Parameters
    ----------
    a21 : float
    Einstein A coefficient for spontaneous emission, s^-1.
    freq : float
    Frequency of transition, Hz. Default=IR transition in Tsuji et al.

    Returns
    -------
    b21 : float
    Einstein B coefficient for stimulated emission, m^2 J^-1 s^-1.
    '''
    b21 = a21 * atm.c_light**2 / (8. * m.pi * atm.h * freq**3)
    return b21
    # account for degeneracies?!?

def b12(a21, g1, g2, freq=nu12):
    '''
    Calculate Einstein coefficient for absorption from A coefficient.

    Parameters
    ----------
    a21: float
    Einstein A coefficient for spontaneous emission, s^-1.
    g1, g2: float
    Degeneracies of lower and upper states.
    freq : float
    Frequency of transition, Hz. Default=IR transition in Tsuji et al.

    Returns
    -------
    b12 : float
    Einstein B coefficient for absorption, m^2 J^-1 s^-1.
    '''
    b12 = g1/g2 * b21(a21, freq)
    return b12

def fwhm_doppler(nu, temp, mass):
    '''
    Calculates FWHM, Hz, of peak from Doppler broadening.

    Source: Demtroder 3rd ed., p 70 (Eqn 3.43a)
    '''
    fwhm = (2*m.pi*nu/c) * (8*kb*temp*m.log(2)/mass)**0.5
    return fwhm

def press_broaden(press=op_press):
    '''
    Uses HITRAN parameters to calculate HWHM, in MHz, at 'press'
    in torr. Default pressure of op_press
    '''
    hwhm = press * atm.torr_to_atm * (g_air) * atm.wavenum_to_Hz / 1e6
    return hwhm

def sat(bandwidth, beam, q=quenchb):
    '''
    Calculates power when saturation parameter is equal to 1,
    following Daily et al., 1977, i.e., when population of excited
    state is half that of ground state, scaled by degeneracies. Full
    saturation is when this parameter is >>1.

    Parameters
    ----------
    bandwidth : float
    Bandwidth of laser, MHz
    beam : float
    Beam diameter, mm
    q : float
    combined quenching and spontaneous emission rate (Q + A), s^-1.
    Default value is literature OH v"=1 --> v"=0 quench rate.

    Returns
    -------
    sat_power : float
    Power when saturation parameter is 1, W.
    '''
    
    sat_intensity = q / b21(a21)
    beam_area = m.pi * (0.5 * beam / 1.e3)**2
    sat_power = sat_intensity * beam_area * (bandwidth * 1.e6)
        # power is I * A * bandwidth

    return sat_power

################################

## Executes when running from within module

if __name__ == "__main__":

    print('%g' % peak_power_UV)