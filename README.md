popmodel
========
Python package to calculate the population of molecules in particular states using a master equation approach. Designed for (and currently only usable) for excitation of the hydroxyl radical (OH) to one excited vibrational state and one excited electronic state, with two different lasers.

This represents a slow accumulation of calculations I've needed to do for a research project, and would need some work to become more generalized.

Capabilities:
- Extract absorption feature information (upper/lower states, energy gap, degeneracies, Einstein coefficients, ...) from a HITRAN-type file. (Vibrational excitation only)
- Calculate shape of absorption feature from Doppler and pressure broadening.
- Automatically define fast modulation of narrow laser linewidth over broadened absorption feature, to determine efficacy of laser modulation to broaden linewidth.
- Solve system of ODEs to calculate population in each state over time. Processes included in the ODEs are stimulated absorption/emission, spontaneous emission, and rotational/vibrational/electronic relaxation.
- Plot populations or laser frequency over time; plot vibrational excitation absorption feature.

The core of popmodel is the KineticsRun object. Each KineticsRun instance includes:
- one Sweep object (upon creation), defining modulation of the laser causing a vibrational excitation
- one Abs object (made after a HITRAN line has been picked), defining the shape of the absorption feature
- other parameters defining operating conditions (temp, pressure, ...) and what processes are included in the ODE

Basic example:

    from popmodel.main import KineticsRun
    # set up KineticsRun with default parameters
    k=KineticsRun()
    # solve ODE
    k.solveode(file='13_hit12.par')
    # plot result
    k.plotpops()
