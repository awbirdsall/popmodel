# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 10:49:58 2014

@author: Adam Birdsall

Kinetic model for two-photon OH LIF with focus on examining IR transition.

Capabilities:

- model population distribution across frequency space for v"=1 <-- v"=0
- model different options for sweeping IR laser freq over time
- use loadhitran to extract parameters from HITRAN file
- collect other physical and experimental parameters from ohcalcs
- integrate ODE describing population in quantum states
- consider populations both within and without rotational level of interest.
- configurable set-up (with/without UV and IR lasers, with/without full
  population output)

"""

from . import ohcalcs as oh
from . import atmcalcs as atm
from . import loadhitran as loadhitran
from . import sweep as sw
from . import absprofile as ap

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from scipy.stats import norm
from math import floor, log
import logging
import yaml
# Prefer CLoader to load yaml, see https://stackoverflow.com/q/18404441
# But, probably makes absolutely no difference for small yaml file used here.
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

##############################################################################
# Set up logging, following python logging cookbook.
# Need to getLogger() here AND in each class/submodule.
# Any logger in form 'popmodel.prefix' will inherit behavior from this logger.
LOGGER = logging.getLogger('popmodel')
LOGGER.setLevel(logging.WARNING)
def add_streamhandler():
    '''Add StreamHandler to LOGGER.
    '''
    LOGGER.setLevel(logging.INFO)
    streamhandler = logging.StreamHandler()
    streamhandler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    streamhandler.setFormatter(formatter)
    LOGGER.addHandler(streamhandler)

def add_filehandler(logfile):
    '''Add FileHandler to LOGGER, with output to `logfile`.
    '''
    filehandler = logging.FileHandler(logfile)
    filehandler.setLevel(logging.INFO)
    logfile_formatter = logging.Formatter('%(asctime)s:%(levelname)s:'
                                          '%(name)s:%(message)s')
    filehandler.setFormatter(logfile_formatter)
    LOGGER.addHandler(filehandler)

def importyaml(parfile):
    '''Extract nested dict of parameters from yaml file.

    See sample parameters.yaml file for full structure. Top-level keys are
    irlaser,sweep,uvlaser,odepar,irline,uvline,detcell,rates
    '''
    with open(parfile, 'r') as yamlfile:
        par = yaml.load(yamlfile, Loader=Loader)
    return par

def automate(hitfile, parameters, logfile=None, csvout=None, image=None,
             verbose=False):
    '''Accept command-line-mode-style inputs and provide requested output(s).
    '''
    if verbose:
        add_streamhandler()
    if logfile:
        add_filehandler(logfile)
    # log each output filename
    argdict = {"log file": logfile, "output csv": csvout,
               "output png image":image}
    for (outputtype, outputfile) in argdict.iteritems():
        if outputfile is not None:
            LOGGER.info('saving '+outputtype+' to '+outputfile)

    par = importyaml(parameters)
    hpar = loadhitran.processhitran(hitfile)
    kineticsrun = KineticsRun(hpar, **par)
    kineticsrun.solveode()
    if csvout:
        kineticsrun.savecsv(csvout)
    if image:
        kineticsrun.popsfigure().savefig(image)

##############################################################################

class KineticsRun(object):
    '''Full model of OH population kinetics: laser, feature and populations.

    If IR laser is swept, has single instance of Sweep, describing IR laser
    dithering. Also has two AbsProfile instances, irfeat and uvfeat, describing
    absorption features. All are made on initialization of KineticsRun
    instance.
    '''
    def __init__(self, hpar, irlaser, sweep, uvlaser, odepar, irline, uvline,
                 detcell, rates):
        '''Initizalize KineticsRun with input recarray (hpar) and dicts (rest).

        The input parameters dicts need to follow the format achieved by taking
        a YAML file of format data/parameters_template.yaml and converting it
        to a list of dicts using popmodel.importyaml. Note that some of these
        dicts are nested. See comments in `parameters_template.yaml` for
        description of individual parameters.

        Parameters:
        -----------
        hpar : recarray
        Describes set of infrared transitions as tabulated in a HITRAN-format
        .par file and extracted using `processhitran` within popmodel's
        `loadhitran` module.

        irlaser : dict
        Parameters describing the infrared laser.

        sweep : dict
        Parameters describing the sweeping (dithering) behavior of the infrared
        laser (if any).

        uvlaser : dict
        Parameters describing the ultraviolet laser.

        odepar : dict
        Parameters describing how the KineticsRun instance represents and
        solves the ODE describing the population dynamics.

        irline : dict
        Parameters describing the particular line the infrared laser is tuned
        to.

        uvline : dict
        Parameters describing the particular line the ultraviolet laser is
        tuned to.

        detcell : dict
        Parameters describing the detection cell in which excitation and
        detection takes place.

        rates : dict
        Parameters describing rates used in calculating population dynamics
        related to excitation and fluorescence.
        '''
        self.logger = logging.getLogger('popmodel.KineticsRun')

        self.detcell = detcell
        # quencher conc
        self.detcell['Q'] = atm.press_to_numdens(self.detcell['press'],
                                                 self.detcell['temp'])
        self.detcell['ohtot'] = self.detcell['Q']*self.detcell['xoh']

        self.irlaser = irlaser
        self.uvlaser = uvlaser
        self.odepar = odepar
        self.uvline = uvline

        # Build dict self.levels of level names, using VIBRONICDICT
        # nomenclature for keys. Each key points to an int corresponding to
        # a particular vibronic state (0 = 'a', 1 = 'b', 2 = 'c', 3 = 'd'),
        # which is the index used by arrays describing population (e.g.,
        # KineticsRun.pop_full)
        self.levels = ({'pi_v0': 0})
        if self.odepar['withoutIR']:
            # either no excitation at all or UV excitation from X(v"=0)
            if self.odepar['withoutUV']:
                pass
            elif self.uvline['vib'][1] == '0':
                # still include X(v"=1) for consistency and for possibility of
                # relaxation to that level
                self.levels.update({'pi_v1': 1})
                self.levels.update({'sigma_v0': 2, 'uv_lower': 0})
                if self.uvline['vib'][0] == '0':
                    self.nlevels = 3
                    self.levels.update({'uv_upper': 2})
                elif self.uvline['vib'][0] == '1':
                    self.nlevels = 4
                    self.levels.update({'sigma_v1': 3, 'uv_upper': 3})
                else:
                    raise ValueError('Unsupported upper v\' of UV transition')
            else:
                raise ValueError('If withoutIR is True, uvline[\'vib\'] must '
                                 'originate in v\"=0')
        else:
            # either no UV excitation or UV excitation from X(v"=1)
            self.levels.update({'pi_v1': 1})
            if self.odepar['withoutUV']:
                pass
            elif self.uvline['vib'][1] == '1':
                # UV excitation into A(v'=0) OR A(v'=1)
                self.levels.update({'sigma_v0': 2, 'uv_lower': 1})
                if self.uvline['vib'][0] == '0':
                    self.nlevels = 3
                    self.levels.update({'uv_upper': 2})
                elif self.uvline['vib'][0] == '1':
                    self.nlevels = 4
                    self.levels.update({'sigma_v1': 3, 'uv_upper': 3})
                else:
                    raise ValueError('Unsupported upper v\' of UV transition')
            else:
                raise ValueError('If withoutIR is False, uvline[\'vib\'] must '
                                 'originate in v\"=1')

        # Sweep object
        if sweep['dosweep']:
            self.dosweep = True
            self.sweep = sw.Sweep(stype=sweep['stype'],
                                  tsweep=sweep['tsweep'],
                                  width=sweep['width'],
                                  binwidth=sweep['binwidth'],
                                  factor=sweep['factor'],
                                  keeptsweep=sweep['keeptsweep'],
                                  keepwidth=sweep['keepwidth'])
            self.sweep.avg_step_in_bin = sweep['avg_step_in_bin']
            # Average number of integration steps to spend in each frequency
            # bin as laser sweeps over frequencies. Default of 20 is
            # conservative, keeps in mind that time in each bin is variable
            # when sweep is sinusoidal.
        else:
            self.dosweep = False
            # stupid hack to be able to use self.sweep.las_bins.size elsewhere
            # and have it be 1.
            self.sweep = lambda: None
            self.sweep.las_bins = np.zeros(1)

        # Instance attributes set up within KineticsRun.solveode():
        # times at which solveode() integration is calculated, s.
        self.tbins = None
        # Index of self.tbins reached by integration progress.
        self.time_progress = None
        # Static array describing sweep in infrared laser frequency with time.
        self.sweepfunc = None
        # compact array of integrated population dynamics, with two entries
        # within each vibronic level (within/without rot level of interest).
        self.pop_abbrev = None
        # full initial array of populations at t=0
        self.pop_init = None
        # full output array of integrated bins
        if self.odepar['keep_pop_full']:
            self.pop_full = None

        # extract invariant kinetics parameters
        self.rates = rates
        # overwrite following subdictionaries for appropriate format for
        # d_pop_full: overall vibrational quenching rate from b:
        self.rates['vet_p1p0']['tot'] = oh.kqavg(rates['vet_p1p0']['n2'],
                                                 rates['vet_p1p0']['o2'],
                                                 rates['vet_p1p0']['h2o'],
                                                 detcell['xh2o'])
        # vibrational quenching rate from c:
        self.rates['kqc']['tot'] = oh.kqavg(rates['kqc']['n2'],
                                            rates['kqc']['o2'],
                                            rates['kqc']['h2o'],
                                            self.detcell['xh2o'])
        self.rates['vet_s1s0']['tot'] = oh.kqavg(rates['vet_s1s0']['n2'],
                                                 rates['vet_s1s0']['o2'],
                                                 rates['vet_s1s0']['h2o'],
                                                 self.detcell['xh2o'])

        # set up IR line
        self.irline = hpar[hpar['label'] == irline][0] # only one label/line
        self.logger.info('__init__: using %s line at %.4g cm^-1',
                         self.irline['label'], self.irline['wnum_ab'])

        # add additional keys to self.uvline
        self._setupuvline()

        # add Einstein B coefficencies to self.rates
        self.rates['Bba'] = self.irline['Bba']
        self.rates['Bab'] = self.irline['Bab']
        # UV Einstein coefficients:
        # Lookup A coefficient based on vibrational band.
        # Have not implemented lookup based on rotational dependence of
        # A coefficient. Crosley and Lengel 1975 says that factor is something
        # like 1-BETA*J"(J"+1) for small J',J" where BETA is on order of 6e-4,
        # meaning correction is on order of 0.99 for J"=3.5. Not an important
        # correction for small J".
        vbc = atm.WAVENUM_TO_HZ*self.uvline['wnum_uv']
        self.rates['Bcb'] = oh.b21(self.rates['A'][self.uvline['vib']], vbc)
        self.rates['Bbc'] = oh.b12(self.rates['A'][self.uvline['vib']],
                                   self.irline['gb'],
                                   self.uvline['g_upper'],
                                   vbc)

        # lookup fractional population in rotational states of interest
        self.rotfrac = self._makerotfracarray()
        # use binwidth to determine resolution of irfeat. Until hard-coded
        # match betwen sweep binwidth and laser bandwidth is teased apart, can
        # only safely dynamically adjust binwidth to give appropriate
        # resolution for lsfactor calculation for given pressure when dosweep
        # is False. Otherwise, preserve resolution  originally passed in.
        if sweep['dosweep']:
            self.binwidth = sweep['binwidth']
        else:
            self.binwidth = 1.e6/760.*detcell['press']

        # set up AbsProfiles and lsfactors
        self.irfeat = self._makeirfeat()
        self.uvfeat = self._makeuvfeat()
        self.irlsfactor = calclsfactor(self.irfeat, self.irlaser)
        self.uvlsfactor = calclsfactor(self.uvfeat, self.uvlaser)

    def _setupuvline(self):
        '''Set up KineticsRun instance parameters related to UV transition.
        '''
        # could refactor this lookup to avoid duplicating same thing in
        # loadhitran.extractnjlabel but whatever...
        br_dict = {'O':-2, 'P':-1, 'Q':0, 'R':1, 'S':2}
        uv_Ndiff = br_dict[self.uvline['rovib'][0]]
        if self.levels.get('uv_lower') == 0:
            self.uvline['Nc'] = self.irline['Na'] + uv_Ndiff
        elif self.levels.get('uv_lower') == 1:
            self.uvline['Nc'] = self.irline['Nb'] + uv_Ndiff
        else: # just need dummy placeholder for _makerotfracarray
            self.uvline['Nc'] = 1
        # same rot level of interest in 'c' and 'd'
        self.uvline['Nd'] = self.uvline['Nc']

        # calculate degeneracy of uv_upper state, 2J+1
        if self.uvline['rovib'][2] == '1':
            j_upper = self.uvline['Nc']+0.5
        elif self.uvline['rovib'][2] == '2':
            j_upper = self.uvline['Nc']-0.5
        else:
            raise ValueError("malformed uvline['rovib']")
        self.uvline['g_upper'] = 2*j_upper + 1

        # lookup uvline wavelength
        if self.odepar['withoutIR']:
            # lower-state energy determined by lower state of (turned-off)
            # infrared transition...
            E_uv_lower = self.irline['E_low']
        else:
            E_uv_lower = self.irline['E_low'] + self.irline['wnum_ab']
        vib_upper = int(self.uvline['vib'][0])
        self.uvline['wnum_uv'] = oh.calculateuv(self.uvline['Nc'],
                                                vib_upper,
                                                int(self.uvline['rovib'][2]),
                                                E_uv_lower)

        vbc = atm.WAVENUM_TO_HZ*self.uvline['wnum_uv']
        if not self.odepar['withoutUV']:
            self.logger.info('setupuvline: using %s line at %d cm^-1 (%.1f nm)',
                             self.uvline['rovib'],
                             self.uvline['wnum_uv'],
                             oh.c/vbc*1e9)

    def _makerotfracarray(self):
        '''Make array of equilibrium rotational pops.

        Requires self.irline and self.uvline to already be created. Array has
        values for a through d vibronic states, regardless of self.nlevels.
        Avoid any issues with size mismatch by zipping through vibronic levels
        and rotfrac arrays as appropriate.
        '''
        # figure out which 'F1' or 'F2' series that a and b state are:
        f_a = int(self.irline['label'][2]) - 1
        if self.irline['label'][3] == '(': # i.e., not '12' or '21' line
            f_b = f_a
        else:
            f_b = int(self.irline['label'][3]) - 1
        rotfrac = np.array([oh.ROTFRAC['a'][f_a][self.irline['Na']-1],
                            oh.ROTFRAC['b'][f_b][self.irline['Nb']-1],
                            oh.ROTFRAC['c'][self.uvline['Nc']],
                            oh.ROTFRAC['d'][self.uvline['Nd']]])
        return rotfrac

    def _makeirfeat(self):
        '''Make an IR absorption profile using self.irline and experimental
        parameters.
        '''
        # Set up IR b<--a absorption profile
        irfeat = ap.AbsProfile(wnum=self.irline['wnum_ab'],
                               binwidth=self.binwidth)
        irfeat.makeprofile(press=self.detcell['press'],
                           T=self.detcell['temp'],
                           g_air=self.irline['g_air'],
                           abswidth=self.detcell['press']*1.e9/300.)
        return irfeat

    def _makeuvfeat(self):
        '''Make a UV absorption profile using self.uvline and self.detcell.
        '''
        uvfeat = ap.AbsProfile(wnum=self.uvline['wnum_uv'],
                               binwidth=self.binwidth*100) # tends to be broader
        uvfeat.makeprofile(press=self.detcell['press'],
                           T=self.detcell['temp'],
                           g_air=self.uvline['g_air'],
                           abswidth=self.detcell['press']*1.e11/300)
        return uvfeat


    def calcfluor(self, timerange=None, duringuvpulse=False):
        '''Calculate total fluorescence counts over given time interval.

        Without any arguments, will look at entire model run.

        Requires KineticsRun.solveode() to have been run, with
        KineticsRun.odepar['keep_pop_full'] = True.

        Parameters
        ----------
        timerange : sequence of floats (length two)
        List of starting time and ending time for calculation.

        duringuvpulse : Boolean
        Whether time interval should span period when UV laser is on. Overrides
        `timerange`.

        Output
        ------
        avgfluorrate : float
        Total fluorescence over time interval, photons.
        '''

        if self.pop_full is None:
            raise AttributeError('KineticsRun instance does not have '
                                 '`pop_full`. Need to have run '
                                 'KineticsRun.solveode() with '
                                 'KineticsRun.odepar[\'keep_pop_full\']=True.')

        # Fluorescence does not apply if the only levels are in the electronic
        # ground state.
        if self.nlevels == 2:
            raise ValueError('calcfluor requires electronically excited pop')

        # define time range
        if duringuvpulse:
            # overwrite any provided timerange argument
            timerange = (self.uvlaser['delay'], (self.uvlaser['delay'] +
                                                 self.uvlaser['pulse']))
        if timerange is not None:
            start = np.searchsorted(self.tbins, timerange[0])
            end = np.searchsorted(self.tbins, timerange[1])
            timerange_s = np.s_[start:end]
        else:
            timerange_s = np.s_[:]

        if self.nlevels == 3 or (self.nlevels == 4 and
                                 self.detcell['fluorwl'] == '308'):
            fluorrates = ['00']
        elif self.nlevels == 4 and self.detcell['fluorwl'] == '282':
            fluorrates = ['10']
        elif self.nlevels == 4 and self.detcell['fluorwl'] == 'both':
            fluorrates = ['00', '10']
        else:
            raise ValueError('fluorescence detection scheme could not be '
                             'determined.')

        fluorcounts = 0.
        for rate in fluorrates:
            fluorlevel = int(rate[0]) + 2 # from A(v'=0)-->index 2, A(v'=1)-->3
            intpop = (self.pop_full[timerange_s, fluorlevel, :].sum()*
                      self.odepar['dt'])
            fluorcounts = fluorcounts + intpop*self.rates['A'][rate]

        return fluorcounts

    def solveode(self):
        '''Integrate ode describing two-photon LIF.

        Use master equation (no Jacobian) and all relevant parameters.

        Define global parameters that are independent of HITRAN OH IR data
        within function: Additional OH parameters related to 'c' state and
        quenching, and laser parameters. Also set up parameters for solving
        and plotting ODE.

        Outputs:
        --------
        pop_full : ndarray
        Relative population of 'a', 'b' (and 'c') states over integration time.
        Three-dimensional array: first dimension time, second dimension a/b/c
        state, third dimension subpopulations within state.

        Subpopulations defined, in order, as (1) bins excited individually by
        swept IR laser (one bin if IR laser sweep off), (2) population in line
        wings not reached by swept IR laser (one always empty bin if IR laser
        sweep off), (3) other half of lambda doublet for a/b PI states, (4)
        other rotational levels within same vibrational level.
        '''

        self.logger.info('solveode: integrating at %d torr, %d K, OH in cell, '
                         '%.2g cm^-3',
                         self.detcell['press'],
                         self.detcell['temp'],
                         self.detcell['ohtot'])
        tl = self.odepar['inttime'] # total int time

        # set-up steps only required if IR laser is swept:
        if self.dosweep:
            self.logger.info('solveode: sweep mode: %s',
                             self.sweep.stype)

            # Align bins for IR laser and absorbance features for integration
            self.sweep.alignbins(self.irfeat)

            # avg_bintime calced for 'sin'. 'saw' twice as long.
            avg_bintime = (self.sweep.tsweep /
                           (2*self.sweep.width/self.sweep.binwidth))
            dt = avg_bintime/self.sweep.avg_step_in_bin
            self.tbins = np.arange(0, tl+dt, dt)
            t_steps = np.size(self.tbins)

            # define local variables for convenience
            num_las_bins = self.sweep.las_bins.size
            # lambda doublet, other rot
            tsweep = self.sweep.tsweep
            stype = self.sweep.stype

            # Determine location of swept IR (a to b) laser by defining 1D
            # array self.sweepfunc: las_bins index for each point in tsweep.
            tindex = np.arange(np.size(self.tbins))
            tindexsweep = np.searchsorted(self.tbins, tsweep, side='right')-1
            if stype == 'saw':
                self.sweepfunc = np.floor((tindex%tindexsweep)*(num_las_bins) /
                                          tindexsweep)
            elif stype == 'sin':
                self.sweepfunc = np.round((num_las_bins-1)/2. *
                                          np.sin(2*np.pi/tindexsweep*tindex) +
                                          (num_las_bins-1)/2.)
            else:
                self.sweepfunc = np.empty(np.size(tindex))
                self.sweepfunc.fill(np.floor(num_las_bins/2))

        else: # single 'bin' excited by laser. Set up in __init__
            dt = self.odepar['dt'] # s
            self.tbins = np.arange(0, tl+dt, dt)
            t_steps = np.size(self.tbins)
            tindex = np.arange(t_steps)
            self.sweepfunc = np.zeros(np.size(tindex))

        self.logger.info('solveode: integrating %.2g s, step size %.2g s',
                         tl, dt)

        # set up ODE

        # Create initial state pop_init, all pop distributed in ground state
        self.pop_init = np.zeros((self.nlevels, self.sweep.las_bins.size+3))
        if self.dosweep:
            p0 = self.detcell['ohtot']
            self.pop_init[0, 0:-3] = (self.irfeat.intpop * self.rotfrac[0] *
                                      p0 / 2)
            # pop outside laser sweep
            self.pop_init[0, -3] = ((self.irfeat.pop.sum() -
                                     self.irfeat.intpop.sum()) *
                                    self.rotfrac[0] *
                                    p0 / 2)
        else:
            if self.irlaser['pulse'] is None:
                # start at steady-state for CW IR laser with UV laser off
                excite = self.ratecoeff('ir_laser', 1.e-9)*self.rates['Bab']
                emit = self.ratecoeff('ir_laser', 1.e-9)*self.rates['Bba']
                laserfrac = 0.5*self.rotfrac[0]
                vetrate = self.rates['vet_p1p0']['tot']*self.detcell['Q']
                ratio = excite*laserfrac/(emit*laserfrac+vetrate)
                p1 = (ratio*self.detcell['ohtot'])/(1 + ratio)
                p0 = self.detcell['ohtot'] - p1
                # set up X(v"=1) with same distribution
                self.pop_init[1, 0] = p1 * self.rotfrac[0] / 2
                self.pop_init[1, -3] = 0
                self.pop_init[1, -2] = p1 * self.rotfrac[0] / 2
                self.pop_init[1, -1] = p1 * (1-self.rotfrac[0])
            else:
                p0 = self.detcell['ohtot']
            self.pop_init[0, 0] = p0 * self.rotfrac[0] / 2
            self.pop_init[0, -3] = 0 # no pop outside laser bandwidth
        # other half of lambda doublet
        self.pop_init[0, -2] = p0 * self.rotfrac[0] / 2
        # other rot
        self.pop_init[0, -1] = p0 * (1-self.rotfrac[0])

        # Create array to store output at each timestep, depending on
        # keep_pop_full:
        # pop_full stores a/b/c state pops in each bin over time.
        # pop_abbrev stores a/b/c pops, tracks in or out rot/lambda of
        # interest.
        if self.odepar['keep_pop_full']:
            self.pop_full = np.empty((t_steps, self.nlevels,
                                      self.sweep.las_bins.size+3))
            self.pop_full[0] = self.pop_init
        else:
            self.pop_abbrev = np.empty((t_steps, self.nlevels, 2))
            self.pop_abbrev[0] = np.array([self.pop_init[:, 0:-2].sum(1),
                                           self.pop_init[:, -2:].sum(1)]).T

        # Initialize scipy.integrate.ode object, lsoda method
        r = ode(self.d_pop_full)
        # r.set_integrator('vode',order=15, nsteps=500,method='bdf')
        r.set_integrator('lsoda', with_jacobian=False,)
        r.set_initial_value(list(self.pop_init.ravel()), 0)

        self.logger.info('  %  |   time   |  bin  ')
        self.logger.info('------------------------')

        # Solve ODE
        self.time_progress = 0 # _laspos looks at this to choose sweepfunc index
        old_complete = 0 # track integration progress for self.logger
        while r.successful() and r.t < tl-dt:
            # display progress
            complete = r.t/tl
            if floor(complete*100/10) != floor(old_complete*100/10):
                self.logger.info(' %3.0f | %8.2g | %4.0f  ',
                                 complete*100,
                                 r.t,
                                 self.sweepfunc[self.time_progress])
            old_complete = complete

            # integrate
            entry = int(round(r.t/dt))+1
            nextstep = r.integrate(r.t + dt)
            nextstep_pop_full = np.resize(nextstep,
                                          (self.nlevels,
                                           self.sweep.las_bins.size + 3))

            # save output
            if self.odepar['keep_pop_full'] == True:
                self.pop_full[entry] = nextstep_pop_full
            else:
                self.pop_abbrev[entry] = np.array([
                    nextstep_pop_full[:, 0:-2].sum(1),
                    nextstep_pop_full[:, -2:].sum(1)
                    ]).T

            self.time_progress += 1

        # Sometimes the solution at the very last time step blows up/crashes to
        # 0. Check that population is conserved and drop last point if
        # necessary. Also avoid any weirdness with nan values popping up by
        # silently converting to 0.
        if self.odepar['keep_pop_full'] == True:
            if not np.isclose(self.pop_full[0].sum(), self.pop_full[-1].sum()):
                self.pop_full = self.pop_full[:-1]
                self.tbins = self.tbins[:-1]
            self.pop_full = np.nan_to_num(self.pop_full)
        else:
            if not np.isclose(self.pop_abbrev[0].sum(),
                              self.pop_abbrev[-1].sum()):
                self.pop_abbrev = self.pop_abbrev[:-1]
                self.tbins = self.tbins[:-1]
            self.pop_abbrev = np.nan_to_num(self.pop_abbrev)

        self.logger.info('solveode: done with integration')

    def _laspos(self):
        '''Determine position of IR laser at current integration time.

        Function of state of self.time_progress, self.sweepfunc and
        self.sweep.las_bins. Only self.time_progress should change over the
        course of an integration in solveode.

        Outputs
        -------
        voigt_pos : int
        Index of self.sweep.las_bins for the frequency that the sweeping laser
        is currently tuned to.
        '''
        voigt_pos = self.sweepfunc[self.time_progress]
        if voigt_pos+1 > self.sweep.las_bins.size:
            self.logger.warning('_laspos: voigt_pos out of range')
        return voigt_pos

    def d_pop_full(self, t, y):
        '''Construct differential equations to describe 2- or 3-state model.

        Parameters:
        -----------
        t : float
        Time
        y: ndarray
        1D-array describing the population in each bin in each energy level.
        Flattened version of multidimensional array `pop_full`.

        Outputs
        -------
        result : ndarray
        1D-array describing d_pop_full in all 'a' states, then 'b', ...
        '''
        # ode method requires y passed in and out of d_pop_full to be
        # one-dimensional. For calculations within d_pop_full, reshape y back
        # into 2D form of pop_full
        y = y.reshape(self.nlevels, -1)

        # calculate fdist and fdist_lambda, distribution *within* single
        # rotational level or lambda level that is relaxed to
        if self.odepar['redistequil']:
            # equilibrium distribution in ground state, as calced for pop_init
            fdist = (self.pop_init[0, :-1] / self.pop_init[0, :-1].sum())
            fdist_lambda = fdist[:-1] / fdist[:-1].sum()
            # for A-state, assume distribution same as pop_init but
            # lambda-doublet is non-existent.
            fdist_A = np.append(fdist_lambda, 0)
            fdist_array = np.vstack((fdist, fdist, fdist_A, fdist_A))
        elif y[0, 0:-1].sum() != 0:
            # instantaneous distribution in ground state
            fdist = (y[0, :-1] / y[0, :-1].sum())
            fdist_lambda = fdist[:-1] / fdist[:-1].sum()
            fdist_A = np.append(fdist_lambda, 0)
            fdist_array = np.vstack((fdist, fdist, fdist_A, fdist_A))
        else:
            fdist_array = np.zeros((y.shape[0], y.shape[1] - 1))

        # generate rates for each process
        # vibronic rates
        vibroniclist = self.vibronicprocesses(y, t)
        vibronicarray = np.sum(vibroniclist.values(), axis=0)

        # rotational equilibration
        rrin = self.rates['rrout'] * self.rotfrac / (1 - self.rotfrac)
        rrates = np.array([self.rates['rrout'], rrin]).T
        if self.odepar['rotequil']:
            rrvalues = np.vstack([
                internalrate(yl, r * self.detcell['Q'], dist, 'rot')
                for yl, r, dist in zip(y, rrates, fdist_array)])
        else:
            rrvalues = np.zeros_like(y)

        # lambda equilibration
        lrin = self.rates['lrout'] # assume equal equilibrium pops
        lrates = np.array([self.rates['lrout'], lrin]).T
        if self.odepar['lambdaequil']:
            lrvalues = np.vstack([internalrate(yl, r * self.detcell['Q'],
                                               fdist_lambda, 'lambda') for
                                  yl, r in zip(y, lrates)])
        else:
            lrvalues = np.zeros_like(y)

        result = vibronicarray + rrvalues + lrvalues
        # flatten to 1D array as required
        return result.ravel()

    def ratecoeff(self, ratetype, t):
        '''Value to multiply base rate by to get first-order rate constant.

        Time-dependent when 'coeff' is pulsed laser intensity. Account for
        extent of lineshape overlap for IR laser with self.lsfactor.
        '''
        if ratetype == 'ir_laser':
            coeff = intensity(t, self.irlaser) * self.irlsfactor
        elif ratetype == 'uv_laser':
            coeff = intensity(t, self.uvlaser) * self.uvlsfactor
        elif ratetype == 'quencher':
            coeff = self.detcell['Q']
        elif ratetype == 'first_order':
            coeff = 1
        else:
            raise ValueError('invalid ratetype')
        return coeff

    def vibronicprocesses(self, y, t):
        '''Create dict of rates for vibronic processes at given instant.

        Builds dict from those processes in VIBRONICDICT that start and end in
        vibronic levels included in KineticsRun instance.

        Parameters
        ----------
        y : ndarray
        Population array used by KineticsRun.d_pop_full

        t : float
        Time (s) used by KineticsRun.d_pop_full and solveode

        Outputs
        -------
        vibronicratedict : dict
        Dictionary of relevant vibronic rates, keyed by name in VIBRONICDICT.
        '''
        vibronicratedict = {process: self.ratearray(process, y, t) for
                            process in VIBRONICDICT if self._included(process)}
        return vibronicratedict

    def ratearray(self, process, y, t):
        '''Calculate rate array for a KineticsRun instance and named process.

        Parameters
        ----------
        process : str
        Name of process in VIBRONICDICT

        y : ndarray
        Population array used by KineticsRun.d_pop_full

        t : float
        Time (s) used by KineticsRun.d_pop_full and solveode

        Outputs
        -------
        rarray : ndarray
        Calculated array of rates, with same shape as y.
        '''
        rarray = np.zeros_like(y)
        startlevel_idx = self._startlevel_lookup(process)
        startrng_idx = self._getrngidx(_getstartrng(process))()
        endlevel_idx = self._endlevel_lookup(process)
        endrng_idx = self._getrngidx(_getendrng(process))()

        base = self._baserate_k(process)
        coeff = self.ratecoeff(_coefftype(process), t)
        conc = y[startlevel_idx, startrng_idx]
        rate = base*coeff*conc
        # for vibronic process, startrng = endrng
        rarray[startlevel_idx, startrng_idx] = -rate
        rarray[endlevel_idx, endrng_idx] = rate
        return rarray

    def _included(self, proc):
        '''Determine whether a VIBRONICDICT process should be included.
        '''
        if _coefftype(proc) == 'ir_laser' and self.odepar['withoutIR'] == True:
            return False
        elif (_coefftype(proc) == 'uv_laser' and
              self.odepar['withoutUV'] == True):
            return False
        else:
            to_include = (_startlevel(proc) in self.levels and
                          _endlevel(proc) in self.levels)
            return to_include

    def _getrngidx(self, rnglabel):
        '''Look up slice for named range within level

        Returns an unbound function because las_bin location is function of
        time.
        '''
        rng = {'las_bin': self._laspos,
               'half_lambda': lambda: np.s_[:-2],
               'rot_level': lambda: np.s_[:-1],
               'rot_other': lambda: np.s_[-1],
               'full': lambda: np.s_[:]}
        return rng[rnglabel]

    def popsfigure(self, title='population dynamics', subpop=None):
        '''For solved KineticsRun, create figure plotting selected subpops.

        Requires `pop_full` as created by `KineticsRun.solveode()`.

        Parameters
        ----------
        title : str
        Title to display at top of plot.

        subpop : list
        List describing subpopulations to plot. Each string in list must be
        a  three-character code of form 'lsn' where l is level (a, b, c, or d),
        s is sublevel ([s]wept, [h]alf of lambda doublet, lambda [d]oublet, or
        entire [l]evel (default) and n is normalization (fraction of lambda
        [d]oublet, [l]evel, entire [p]opulation (default), or [a]bsolute).

        Outputs
        -------
        fig : matplotlib figure
        Figure with each subpop plotted. Subpops in 'a' and 'b' vibronic levels
        are plotted in ax0, with the left y-axis scale. Subpops in 'c' and 'd'
        are plotted in ax1, with the right y-axis scale.
        '''
        if self.pop_full is None:
            raise AttributeError('KineticsRun instance does not have '
                                 '`pop_full`. Need to have run '
                                 'KineticsRun.solveode() with '
                                 'KineticsRun.odepar[\'keep_pop_full\']=True.')

        fig, (ax0) = plt.subplots()

        # default plot
        if subpop is None:
            subpop = ['blp']

        # make twinx if any 'c' or 'd' will be plotted
        maketwinx = any([(s[0] == 'c') or (s[0] == 'd') for s in subpop])
        if maketwinx:
            ax1 = ax0.twinx()
            # prevent wide number tick labels (e.g., "0.00001") from cutting
            # off y axis labels
            fig.subplots_adjust(left=0.2, right=0.8)
            ax1.set_ylabel('sigma state populations (c and/or d)')

        for plotcode in subpop:
            if plotcode[0] == 'a' or plotcode[0] == 'b':
                ax0.plot(self.tbins*1e6, self.popseries(plotcode),
                         label=plotcode)
                if maketwinx:
                    next(ax1._get_lines.prop_cycler)
            elif plotcode[0] == 'c' or plotcode[0] == 'd':
                ax1.plot(self.tbins*1e6, self.popseries(plotcode),
                         label=plotcode)
                next(ax0._get_lines.prop_cycler)
            else:
                raise NameError("improper plotcode ", plotcode)

        ax0.set_title(title)
        ax0.set_xlabel(r'Time ($\mu$s)')
        ax0.set_ylabel('pi state pops (a or b)')
        ax0.set_xlim((0, self.odepar['inttime']*1e6))

        # gather up lines and labels from both axes for unified legend
        lines_list = [ax.get_legend_handles_labels()[0] for ax in
                      fig.get_axes()]
        lines = [items for sublists in lines_list for items in sublists]
        labels_list = [ax.get_legend_handles_labels()[1] for ax in
                       fig.get_axes()]
        labels = [items for sublists in labels_list for items in sublists]

        topax = fig.get_axes()[-1]
        topax.legend(lines, labels)

        return fig

    def popseries(self, plotcode):
        '''calculate 1D array describing timeseries of subpopulation.
        '''
        if self.pop_full is None:
            raise AttributeError('KineticsRun instance does not have '
                                 '`pop_full`. Need to have run '
                                 'KineticsRun.solveode() with '
                                 'KineticsRun.odepar[\'keep_pop_full\']=True.')

        # add default if second or third character of plotcode blank
        if len(plotcode) == 3:
            pass
        elif len(plotcode) == 2:
            plotcode = plotcode + 'p'
        elif len(plotcode) == 1:
            plotcode = plotcode + 'lp'
        else:
            raise NameError("improper plotcode ", plotcode)

        leveldict = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
        levelidx = leveldict[plotcode[0]]
        sublevelabbrevdict = {'s': 'swept',
                              'h': 'lambda_half',
                              'd': 'rot_level',
                              'l': 'full'}
        # use existing SLICEDICT for sublevel
        sublevelslice = SLICEDICT[sublevelabbrevdict[plotcode[1]]]
        # all sublevels used here should be a slice, so num requires .sum(1).
        # however, just in case sublevel slice is single entry, check ndim of
        # numraw to see if .sum(1) is necessary.
        numraw = self.pop_full[:, levelidx, sublevelslice]
        if numraw.ndim == 1:
            num = numraw
        else:
            num = numraw.sum(1)

        denomabbrev = plotcode[2]
        if denomabbrev == 'd':
            denom = self.pop_full[:, levelidx, SLICEDICT['rot_level']].sum(1)
        elif denomabbrev == 'l':
            denom = self.pop_full[:, levelidx, :].sum(1)
        elif denomabbrev == 'p':
            denom = self.detcell['ohtot']
        elif denomabbrev == 'a':
            denom = 1
        else:
            raise NameError("improper plotcode ", plotcode)

        popseries = num/denom
        return popseries

    def vslaserfigure(self, func, title='plot', ylabel='y axis'):
        '''Make arbitrary plot in time with laser sweep as second plot

        Parameters
        ----------
        func : ndarray
        1D set of values that is function of self.tbins

        title : str
        Title to display on top of plot

        ylabel : str
        Y-axis label to display.

        pngout : str
        filename to save PNG output. Displays plot if not given.
        '''
        if self.pop_abbrev is None and self.pop_full is None:
            raise AttributeError('Need to have run KineticsRun.solveode()')

        elif self.pop_abbrev is None and self.odepar['keep_pop_full']:
            self.pop_abbrev = np.empty((np.size(self.tbins), self.nlevels, 2))
            self.pop_abbrev[:, :, 0] = self.pop_full[:, :, 0:-2].sum(2)
            self.pop_abbrev[:, :, 1] = self.pop_full[:, :, -2:].sum(2)

        fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
        fig.subplots_adjust(hspace=.3)
        x_usec = self.tbins*1e6 # microseconds tends to be natural units
        ax0.plot(x_usec, func)
        ax0.set_title(title)
        ax0.set_ylabel(ylabel)

        if self.dosweep:
            time_idx = np.arange(np.size(self.tbins))
            las_idx_from_sweepfunc = self.sweepfunc[time_idx].astype(int)
            y_las_freq_mhz = self.sweep.las_bins[las_idx_from_sweepfunc]/1e6
            ax1.plot(x_usec, y_las_freq_mhz)
        else:
            ax1.plot(x_usec, self.tbins*0)
        ax1.set_title('Position of IR beam')
        ax1.set_xlabel(r'Time ($\mu$s)')
        ax1.set_ylabel('Relative Frequency (MHz)')

        return fig

    def absfigure(self, laslines=True):
        '''Plot the calculated absorption feature in frequency space

        Requires KineticsRun instance with an Abs that makeProfile has been run
        on (i.e., have self.irfeat.abs_freq and self.irfeat.pop)

        Parameters
        ----------
        laslines : Bool
        Whether to plot the edges of where the laser sweeps. Requires the
        KineticsRun instance to have a Sweep with self.sweep.las_bins array.
        '''
        fig, (ax0) = plt.subplots(nrows=1)
        ax0.plot(self.irfeat.abs_freq/1e6, self.irfeat.pop)
        ax0.set_title('Calculated absorption feature, '
                      + str(self.detcell['press']) +' torr')
        ax0.set_xlabel('Relative frequency (MHz)')
        ax0.set_ylabel('Relative absorption')

        if laslines:
            ax0.axvline(self.sweep.las_bins[0]/1e6, ls='--')
            ax0.axvline(self.sweep.las_bins[-1]/1e6, ls='--')
        return fig

    def savecsv(self, csvout):
        '''save csv of 3-level system populations and time values

        first column is time, next three columns are populations of a, b and
        c in state of interest.'''

        if self.pop_abbrev is None and self.pop_full is None:
            raise AttributeError('Need to have run KineticsRun.solveode()')
        elif self.pop_abbrev is None:
            self.pop_abbrev = np.empty((np.size(self.tbins), self.nlevels, 2))
            self.pop_abbrev[:, :, 0] = self.pop_full[:, :, 0:-2].sum(2)
            self.pop_abbrev[:, :, 1] = self.pop_full[:, :, -2:].sum(2)
        timeseries = self.tbins[:, np.newaxis] # bulk out so ndim = 2
        pop_abbrev_slice = self.pop_abbrev[:, :, 0] # include states of
                                                    # interest
        np.savetxt(csvout, np.hstack((timeseries, pop_abbrev_slice)),
                   delimiter=",", fmt="%.6e")

    def saveoutput(self, npzfile):
        '''Save result of solveode to npz file.

        Saves arrays describing the population over time, laser bins, tbins,
        sweepfunc, absorption frequencies, and Voigt profile.

        Parameters
        ----------
        file : str
        Path of file to save output (.npz extension standard).
        '''

        if self.pop_abbrev is None and self.pop_full is None:
            raise AttributeError('Need to have run KineticsRun.solveode()')
        elif self.pop_abbrev is None:
            self.pop_abbrev = np.empty((np.size(self.tbins), self.nlevels, 2))
            self.pop_abbrev[:, :, 0] = self.pop_full[:, :, 0:-2].sum(2)
            self.pop_abbrev[:, :, 1] = self.pop_full[:, :, -2:].sum(2)

        np.savez(npzfile,
                 pop_abbrev=self.pop_abbrev,
                 las_bins=self.sweep.las_bins,
                 tbins=self.tbins,
                 sweepfunc=self.sweepfunc,
                 abs_freq=self.irfeat.abs_freq,
                 pop=self.irfeat.pop)

    def loadoutput(self, npzfile):
        '''Populate KineticsRun instance with results saved to npz file.

        Writes to values for pop_abbrev, sweep.las_bins, tbins, sweepfunc,
        irfeat, irfeat.abs_freq and irfeat.pop.

        Parameters
        ----------
        file : str
        Path of npz file with saved output.
        '''
        with np.load(npzfile) as data:
            self.pop_abbrev = data['pop_abbrev']
            self.sweep.las_bins = data['las_bins']
            self.tbins = data['tbins']
            self.sweepfunc = data['sweepfunc']
            self.irfeat = ap.AbsProfile(wnum=self.irline['wnum_ab'])
            self.irfeat.abs_freq = data['abs_freq']
            self.irfeat.pop = data['pop']

    # interpret ratetype parameters in terms of particular KineticsRun instance
    def _baserate_k(self, ratetype):
        '''Look up rate constant for ratetype within KineticsRun.rates
        '''
        return _getnested(_baserate(ratetype), self.rates)
    def _startlevel_lookup(self, ratetype):
        '''Look up initial vibronic level index for ratetype.
        '''
        return self.levels[_startlevel(ratetype)]
    def _endlevel_lookup(self, ratetype):
        '''Look up final vibronic level index for ratetype.
        '''
        return self.levels[_endlevel(ratetype)]

def _getnested(keys, nesteddict):
    '''Access data in nested dict using a list of keys.'''
    if not isinstance(keys, list): # handle single key
        keys = [keys]
    return reduce(dict.__getitem__, keys, nesteddict)

###############################################################################
# GLOBAL VARS
SLICEDICT = {'rot_level': np.s_[:-1],
             'rot_other': np.s_[-1],
             'lambda_half': np.s_[:-2],
             'lambda_other': np.s_[-2],
             'swept': np.s_[:-3],
             'half_lambda': np.s_[:-2],
             'full': np.s_[:]}
# Form of VIBRONICDICT:
# rate constant, 'coefftype' it needs to be multiplied by (or `'first_order' if
# first-order), initial level, final level, initial range (within level), final
# range.  Name each entry with description of type of process, followed by
# suffix that describes the transition: 'ir' or 'uv' if laser transition, 'p0',
# 'p1', 's0', 's1' for v=0 or 1 level of pi (X) or sigma (A) state,
# respectively.
VIBRONICDICT = {'absorb_ir':['Bab',
                             'ir_laser',
                             'pi_v0',
                             'pi_v1',
                             'las_bin',
                             'las_bin'],
                'absorb_uv':['Bbc',
                             'uv_laser',
                             'uv_lower',
                             'uv_upper',
                             'half_lambda',
                             'half_lambda'],
                'stim_emit_ir':['Bba',
                                'ir_laser',
                                'pi_v1',
                                'pi_v0',
                                'las_bin',
                                'las_bin'],
                'stim_emit_uv':['Bcb',
                                'uv_laser',
                                'uv_upper',
                                'uv_lower',
                                'half_lambda',
                                'half_lambda'],
                # 'spont_emit_p1p0':['Aba',
                #                  None,
                #                  'pi_v1',
                #                  'pi_v0',
                #                  'full',
                #                  'full'],
                'vib_quench_p1p0':[['vet_p1p0', 'tot'],
                                   'quencher',
                                   'pi_v1',
                                   'pi_v0',
                                   'full',
                                   'full'],
                'elec_quench_s0p0':[['kqc', 'tot'],
                                    'quencher',
                                    'sigma_v0',
                                    'pi_v0',
                                    'full',
                                    'full'],
                'elec_quench_s1p0':[['kqc', 'tot'],
                                    'quencher',
                                    'sigma_v1',
                                    'pi_v0',
                                    'full',
                                    'full'],
                'spont_emit_s0p0':[['A', '00'],
                                   'first_order',
                                   'sigma_v0',
                                   'pi_v0',
                                   'full',
                                   'full'],
                'spont_emit_s0p1':[['A', '01'],
                                   'first_order',
                                   'sigma_v0',
                                   'pi_v1',
                                   'full',
                                   'full'],
                'spont_emit_s1p0':[['A', '10'],
                                   'first_order',
                                   'sigma_v1',
                                   'pi_v0',
                                   'full',
                                   'full'],
                'spont_emit_s1p1':[['A', '11'],
                                   'first_order',
                                   'sigma_v1',
                                   'pi_v1',
                                   'full',
                                   'full'],
                'vib_quench_s1s0':[['vet_s1s0', 'tot'],
                                   'quencher',
                                   'sigma_v1',
                                   'sigma_v0',
                                   'full',
                                   'full']}
# accessors for VIBRONICDICT
def _baserate(ratetype):
    '''Access string of rate constant for ratetype in VIBRONICDICT.
    '''
    return VIBRONICDICT[ratetype][0]
def _coefftype(ratetype):
    '''Access string of coefficient for ratetype in VIBRONICDICT.
    '''
    return VIBRONICDICT[ratetype][1]
def _startlevel(ratetype):
    '''Access string of starting vibronic level for ratetype in VIBRONICDICT.
    '''
    return VIBRONICDICT[ratetype][2]
def _endlevel(ratetype):
    '''Access string of ending vibronic level for ratetype in VIBRONICDICT.
    '''
    return VIBRONICDICT[ratetype][3]
def _getstartrng(ratetype):
    '''Access range within startlevel for ratetype in VIBRONICDICT.
    '''
    return VIBRONICDICT[ratetype][4]
def _getendrng(ratetype):
    '''Access range within endlevel for ratetype in VIBRONICDICT.
    '''
    return VIBRONICDICT[ratetype][5]

def intensity(timearray, laser):
    '''Calculate spec intensity of laser at given array of times.

    Converts provided power to peak power if the laser is pulsed.'''
    timearray = np.asarray(timearray)
    scalar_input = False
    if timearray.ndim == 0:
        timearray = timearray[None] # convert to 1D
        scalar_input = True
    intensities = np.zeros_like(timearray)
    area = np.pi*(laser['diam']*0.5)**2
    if not laser['pulse']:
        spec_intensity = oh.spec_intensity(laser['power'],
                                           area,
                                           laser['bandwidth'])
        intensities.fill(spec_intensity)
    elif laser['reprate'] is not None:
        whenon = ((timearray%laser['reprate'] > laser['delay']) &
                  (timearray%laser['reprate'] < laser['pulse']+laser['delay']))
        peakpower = oh.peakpower(laser['power'],
                                 laser['pulse'],
                                 laser['reprate'])
        spec_intensity = oh.spec_intensity(peakpower,
                                           area,
                                           laser['bandwidth'])
        intensities[whenon] = spec_intensity
    else:
        raise ValueError('malformed laser parameters')

    if scalar_input:
        return np.squeeze(intensities)
    return intensities

def internalrate(ylevel, ratecon, equildist, ratetype):
    '''calculate contribution to overall rate array for process
    internal to single a/b/c level with equilibrium distribution.

    Parameters
    ----------
    ylevel : array
    1D array of population in single a/b/c level. Assumes consistent format of
    (bins within lambda doublet), lambda doublet, other rotational levels.
    ratecon : list (length 2)
    First-order rate constants for forward and reverse process, s^-1
    equildist : float
    Equilibrium distribution of process
    ratetype : str ('rot' or 'lambda')
    Type of process: rotational or lambda relaxation.

    Output
    ------
    term : np.ndarray
    2D array shaped like y containing rates for process
    '''
    term = np.zeros_like(ylevel)
    if ratetype == 'rot':
        rngin = np.s_[0:-1]
        rngout = np.s_[-1]
    elif ratetype == 'lambda':
        rngin = np.s_[0:-2]
        rngout = np.s_[-2]
    else:
        raise ValueError('ratetype needs to be \'rot\' or \'lambda\'')

    if ylevel[rngin].sum() != 0:
        term[rngin] = (-ylevel[rngin] * ratecon[0] +
                       ylevel[rngout] * ratecon[1] * equildist)
        term[rngout] = (ylevel[rngin].sum() * ratecon[0] -
                        ylevel[rngout] * ratecon[1])
    else:
        term.fill(0)
    return term

def calclsfactor(line, laser):
    '''Calculate factor in laser excitation rate due to lineshapes.

    Assumes the peaks of the two lineshapes are perfectly aligned.

    Treat factor as separate from the inherent "intensity" of the laser
    calculated by the `intensity` function. Instead, it modulates the
    intensity in `ratecoeff`.

    See notes and 15_linewidth_factor_laser_excitation.ipynb for detailed
    derivation of factor. Requires integral of product of normalized
    absorption feature and laser lineshape profiles.'''
    # calculate Gaussian IR lineshape, converting from FWHM to std dev.
    lasershape = norm.pdf(line.abs_freq,
                          scale=laser['bandwidth']/(2*(2*log(2))**0.5))
    # line.pop includes the frequency differential baked in
    integral = (lasershape * line.pop).sum()
    # laser bandwidth term "undoes" the spectral part of the spectral
    # intensity calculation
    lsfactor = integral * laser['bandwidth']
    return lsfactor
