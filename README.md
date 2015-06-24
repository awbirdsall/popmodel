popmodel
========
Python package to calculate the population of molecules in particular states using a master equation approach. Designed for (and currently only usable) for excitation of the hydroxyl radical (OH) to one excited vibrational state and one excited electronic state, with two different lasers.

This represents a slow accumulation of calculations I've needed to do for a research project, and would need some work to become more generalized.

Capabilities:
- Extract absorption feature information (upper/lower states, energy gap, degeneracies, Einstein coefficients, ...) from a HITRAN-type file using `loadHITRAN`. (Vibrational lines of OH only, with limited parsing of H2O.)
- Calculate shape of absorption feature from Doppler and pressure broadening.
- Automatically define fast modulation of narrow laser linewidth over broadened absorption feature.
- Solve system of ODEs to calculate population in each state over time. Processes included in the ODEs are stimulated absorption/emission, spontaneous emission, and rotational/vibrational/electronic relaxation.
- Plot populations or laser frequency over time; plot vibrational excitation absorption feature.

The core of popmodel is the KineticsRun object. Each KineticsRun instance includes:
- one Sweep object (upon creation, if IR laser is being dithered), defining modulation of the laser causing a vibrational excitation
- one Abs object (made after a HITRAN line has been picked, again only if IR laser dithering is turned on), defining the shape of the absorption feature
- other parameters defining operating conditions (lasers, temp, pressure, ...) and ODE solver behavior. Constants and formulas related to OH spectroscopy are collected in the `ohcalcs` module. Experimental parameters are passed into the KineticsRun object as a set of dictionaries when first defined. The expected dictionary format is designed for extraction from a YAML file with command line use.

popmodel can be run from an interactive python session or from the command line.
