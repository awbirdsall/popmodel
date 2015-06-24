# popmodel
Python package to calculate the population of molecules in particular quantum states using a master equation approach. Designed for (and currently only usable) for excitation of the hydroxyl radical (OH) to one excited vibrational state and one excited electronic state, with two different lasers.

This represents a slow accumulation of calculations I've needed to do for a research project, and would need some work to become more generalized.

## Capabilities
- Extract absorption feature information (upper/lower states, energy gap, degeneracies, Einstein coefficients, ...) from a HITRAN-type file using `loadHITRAN`. (Vibrational lines of OH only, with limited parsing of H2O.)
- Calculate shape of absorption feature from Doppler and pressure broadening (`main.Abs` object).
- Automatically define fast modulation of narrow laser linewidth over broadened absorption feature (`main.Sweep` object).
- Solve system of ODEs to calculate population in each state over time. Processes included in the ODEs are stimulated absorption/emission, spontaneous emission, and lambda doublet/rotational/vibrational/electronic relaxation (`main.KineticsRun` object).
- Plot populations or laser frequency over time; plot vibrational excitation absorption feature (`main.KineticsRun.plot[...]()` functions).
- convenience unit conversion functions related to atmospheric science (`atmcalcs`)
- constants and functions related to OH spectroscopy (`ohcalcs`)

The core of `popmodel` is the `KineticsRun` object. Each `KineticsRun` instance requires dictionaries of parameters describing rates of spectroscopic transitions, lasers, detection cell, transition lines, and ODE integration.  The expected dictionary format is designed for extraction from a YAML file and compatible with command line use.

## Required input files

### Hitran file
Infrared line parameters are extracted from the 140-character-format HITRAN 2012 file for OH (default filename `13_hit12.par`), which can be accessed at https://www.cfa.harvard.edu/HITRAN/. Some low-level functions within `loadHITRAN` module can also read other molecules' HITRAN files, but trying to go through the full workflow called by `loadHITRAN.processHITRAN()` used in setting up a `KineticsRun` will not work due to the need to parse strings describing molecule-specific term descriptions. See the HITRAN website for more documentation related to the record format.

### YAML parameter file
Parameters for setting up a `KineticsRun` instance are organized in dictionaries corresponding to a YAML parameter file. A template for the format that the YAML file must follow can be found at `src/popmodel/data/parameters_template.yaml`.

## Example usage

### Command line
Installation using `pip` creates command-line command `popmodel`. Format of command line arguments: `HITFILE PARAMETERS [-l] LOGFILE [-c] CSVOUTPUT [-i] IMAGE`

For example:

~~~
popmodel 13_hit12.par parameters.yaml -l output.log -c output.csv -i output.png
~~~

### Python session

Basic usage:

~~~
import popmodel as pm
par = pm.importyaml("path_to/yaml/parameters.yaml")
hpar = pm.loadHITRAN.processHITRAN("path_to/13_hit12.par")
k = pm.KineticsRun(hpar,**par)
k.solveode()
k.plotpops()
~~~

## Installation
`pip install popmodel` install from PyPI

`pip install git+https://github.com/awbirdsall/popmodel` installs most recent commit on github (bleeding-edge)

## Dependencies
Written for Python 2.7. Requires `numpy`, `scipy`, `pandas`, `pyyaml` and `matplotlib` (automatically handled using `pip` to install). Developed on Windows 64-bit.
