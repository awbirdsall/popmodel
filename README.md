# popmodel
Python package to calculate the population of molecules in particular quantum states using a master equation approach. Designed for (and currently only usable) for excitation of the hydroxyl radical (OH) to one excited vibrational state and one excited electronic state, with two different lasers, and up to four vibronic states being tracked.

This represents a slow accumulation of calculations I've needed to do for a research project, and would need some work to become more generalized.

## Capabilities
- Extract absorption feature information (upper/lower states, energy gap, degeneracies, Einstein coefficients, ...) from a HITRAN-type file using `loadhitran`. (Vibrational lines of OH only, with limited parsing of H2O.)
- Calculate shape of absorption feature from Doppler and pressure broadening (`absprofile.AbsProfile` object).
- Automatically define fast modulation of narrow laser linewidth over broadened absorption feature (`sweep.Sweep` object).
- Solve system of ODEs to calculate population in each state over time. Processes included in the ODEs are stimulated absorption/emission, spontaneous emission, and lambda doublet/rotational/vibrational/electronic relaxation (`main.KineticsRun` object).
- Create `matplotlib` figures of populations or laser frequency over time; create figure of infrared absorption feature (`main.KineticsRun.[...]figure()` functions).
- convenience unit conversion functions related to atmospheric science (`atmcalcs`)
- constants and functions related to OH spectroscopy (`ohcalcs`)

The core of `popmodel` is the `KineticsRun` object. Each `KineticsRun` instance requires dictionaries of parameters describing rates of spectroscopic transitions, lasers, detection cell, transition lines, and ODE integration.  The expected dictionary format is designed for extraction from a YAML file and compatible with command line use.

## Required input files

### Hitran file
Infrared line parameters are extracted from the 140-character-format HITRAN 2012 file for OH (default filename `13_hit12.par`), which can be accessed at https://www.cfa.harvard.edu/HITRAN/. Some low-level functions within `loadhitran` module can also read other molecules' HITRAN files, but trying to go through the full workflow called by `loadhitran.processhitran()` used in setting up a `KineticsRun` will not work due to the need to parse strings describing molecule-specific term descriptions. See the HITRAN website for more documentation related to the record format.

An 200-line excerpt from the OH HITRAN file is included at `src/popmodel/data/hitran_sample.par` for use by the test module. To extract the path to `hitran_sample.par`:

~~~
from pkg_resources import resource_filename
hpath = resource_filename('popmodel','data/hitran_sample.par')
~~~

### YAML parameter file
Parameters for setting up a `KineticsRun` instance are organized in dictionaries corresponding to a YAML parameter file. A template for the format that the YAML file must follow can be found at `src/popmodel/data/parameters_template.yaml`.

To extract the path to `parameters_template.yaml` within `popmodel`:

~~~
from pkg_resources import resource_filename
yamlpath = resource_filename('popmodel','data/parameters_template.yaml')
~~~

## Example usage

### Command line
Installation using `pip` creates command-line command `popmodel`. Format of command line arguments: `HITFILE PARAMETERS [-l] LOGFILE [-c] CSVOUTPUT [-i] IMAGE [-v]`

For example:

~~~
popmodel 13_hit12.par parameters.yaml -l output.log -c output.csv -i output.png
~~~

### Python session

Basic usage:

~~~
import popmodel as pm
pm.add_streamhandler() # optional, print logging.INFO to screen
pm.add_filehandler("path/to.log") # optional, write logging.INFO to file
par = pm.importyaml("path_to/yaml/parameters.yaml")
hpar = pm.loadhitran.processhitran("path_to/13_hit12.par")
k = pm.KineticsRun(hpar,**par)
k.solveode()
k.popsfigure()
~~~

## Installation
`pip install popmodel` install from PyPI

`pip install git+https://github.com/awbirdsall/popmodel` installs most recent commit on github (bleeding-edge is `develop` branch)

## Dependencies
Written for Python 2.7. Requires `numpy`, `scipy`, `pandas`, `pyyaml` and `matplotlib` (automatically handled using `pip` to install). Tests written using `pytest` using the [`pytest-mpl` plugin](https://github.com/astrofrog/pytest-mpl). Developed in a Windows environment.
