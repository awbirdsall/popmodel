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
- turn off UV laser calculations an option to save memory

"""

# modules within package
from . import ohcalcs as oh
from . import atmcalcs as atm
from . import loadhitran as loadhitran
from . import sweep as sw
from . import absprofile as ap

# other modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from math import floor
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
logger = logging.getLogger('popmodel')
logger.setLevel(logging.WARNING)
def add_streamhandler():
    '''Add StreamHandler `ch` to logger.
    '''
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def add_filehandler(logfile):
    '''Add FileHandler `fh` to logger, with output to `logfile`.
    '''
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.INFO)
    logfile_formatter = logging.Formatter('%(asctime)s:%(levelname)s:'+
        '%(name)s:%(message)s')
    fh.setFormatter(logfile_formatter)
    logger.addHandler(fh)

def importyaml(parfile):
    '''Extract nested dict of parameters from yaml file.

    See sample parameters.yaml file for full structure. Top-level keys are
    irlaser,sweep,uvlaser,odepar,irline,uvline,detcell,rates
    '''
    with open(parfile, 'r') as f:
        par = yaml.load(f,Loader=Loader) 
    return par

def automate(hitfile,parameters,logfile=None,csvout=None,image=None,
             verbose=False):
    '''Accept command-line-mode-style inputs and provide requested output(s).
    '''
    if verbose:
        add_streamhandler()
    if logfile:
        add_filehandler(logfile)
    # log each output filename
    argdict = {"log file":logfile,"output csv":csvout,
            "output png image":image}
    for (k,v) in argdict.iteritems():
        if v:
            logger.info('saving '+k+' to '+v)

    par = importyaml(parameters)
    hpar = loadhitran.processhitran(hitfile)
    k = KineticsRun(hpar,**par)
    k.solveode()
    if csvout:
        k.savecsv(csvout)
    if image:
        k.popsfigure().savefig(image)

##############################################################################

class KineticsRun(object):
    '''Full model of OH population kinetics: laser, feature and populations.
    
    If IR laser is swept, has single instance of Sweep, describing laser
    dithering, and of AbsProfile, describing absorption feature. Sweep is made
    in __init__, while AbsProfile is made after the HITRAN file is imported and
    the absorption feature selected.
    '''
    def __init__(self, hpar, irlaser, sweep, uvlaser, odepar, irline, uvline,
                 detcell, rates):
        '''Initizalize KineticsRun using dictionaries of input parameters and
        processed HITRAN file.
        
        The input parameters can be gathered up in a yaml file (in format of
        parameters.yaml) and passed in from the command line.
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
        self.irline = irline
        self.uvline = uvline

        # Build dictionary of level names, based on instance-independent names
        # in `levels`
        self.levelsidx = levels

        if self.odepar['withoutUV']:
            self.nlevels = 2
        else:
            self.levelsidx.update({'uv_lower': 1})
            if self.uvline['vib'][0]=='0':
                self.nlevels = 3
                self.levelsidx.update({'uv_upper': 2})
            else:
                self.nlevels = 4
                self.levelsidx.update({'uv_upper': 3})

        self.system = buildsystem(self.nlevels)

        # Sweep object
        if sweep['dosweep']:
            self.dosweep=True
            self.sweep=sw.Sweep(stype=sweep['stype'],
                                tsweep=sweep['tsweep'],
                                width=sweep['width'],
                                binwidth=sweep['binwidth'],
                                factor=sweep['factor'],
                                keepTsweep=sweep['keepTsweep'],
                                keepwidth=sweep['keepwidth'])
            self.sweep.avg_step_in_bin = sweep['avg_step_in_bin']
            # Average number of integration steps to spend in each frequency
            # bin as laser sweeps over frequencies. Default of 20 is
            # conservative, keeps in mind that time in each bin is variable
            # when sweep is sinusoidal.
        else:
            self.dosweep=False
            # stupid hack to be able to use self.sweep.las_bins.size elsewhere
            # and have it be 1.
            self.sweep = lambda: None
            self.sweep.las_bins = np.zeros(1)
        
        # extract invariant kinetics parameters
        self.rates = rates
        # overwrite following subdictionaries for appropriate format for dN:
        # overall vibrational quenching rate from b:
        self.rates['kqb']['tot'] = oh.kqavg(rates['kqb']['n2'],
                                            rates['kqb']['o2'],
                                            rates['kqb']['h2o'],
                                            detcell['xh2o'])
        # vibrational quenching rate from c:
        self.rates['kqc']['tot'] = oh.kqavg(rates['kqc']['n2'],
                                            rates['kqc']['o2'],
                                            rates['kqc']['h2o'],
                                            self.detcell['xh2o'])
        self.chooseline(hpar,irline)

    def chooseline(self,hpar,label):
        '''Save single line of processed HITRAN file to self.hline.
        '''
        lineidx = np.where(hpar['label']==label)[0][0]
        self.hline = hpar[lineidx]
        self.logger.info('chooseline: using {} line at {:.4g} cm^-1'
                         .format(self.hline['label'], self.hline['wnum_ab']))

        # extract rotation fraction for a b and c
        # figure out which 'F1' or 'F2' series that a and b state are:
        f_a = int(self.hline['label'][2]) - 1
        if self.hline['label'][3] == '(': # i.e., not '12' or '21' line
            f_b = f_a
        else:
            f_b = int(self.hline['label'][3]) - 1
        self.rotfrac = np.array([oh.rotfrac['a'][f_a][self.hline['Na']-1],
                                 oh.rotfrac['b'][f_b][self.hline['Nb']-1],
                                 oh.rotfrac['c'][self.hline['Nc']],
                                 # TODO refactor 'Nc" out of hline
                                 # assume Nd = Nc
                                 oh.rotfrac['d'][self.hline['Nc']]])
        self.rates['Bcb'] = self.hline['Bcb']
        self.rates['Bbc'] = self.hline['Bbc']
        self.rates['Bba'] = self.hline['Bba']
        self.rates['Bab'] = self.hline['Bab']

    def makeAbs(self):
        '''Make an absorption profile using self.hline and experimental
        parameters.
        '''
        # Set up IR b<--a absorption profile
        self.abfeat = ap.AbsProfile(wnum=self.hline['wnum_ab'])
        self.abfeat.makeprofile(press=self.detcell['press'],
                                T=self.detcell['temp'],    
                                g_air=self.hline['g_air'])

    def calcfluor(self, timerange = None, duringuvpulse = False):
        '''Calculate average fluorescence (photons/s) over given time interval.

        Parameters
        ----------
        timerange : sequence of floats (length two)
        List of starting time and ending time for calculation.

        duringuvpulse : Boolean
        Define time interval to span period when UV laser is on. Overrides
        `timerange` argument.

        Output
        ------
        avgfluorescencerate : float
        Average fluorescence rate over time interval, photons/s.
        '''
        if self.nlevels == 2:
            return None

        # define time range
        if duringuvpulse:
            timerange = (self.uvlaser['delay'], (self.uvlaser['delay']
                                                 + self.uvlaser['pulse']))
        if timerange is not None:
            dt = timerange[1]-timerange[0]
            start = np.searchsorted(self.tbins, timerange[0])
            end = np.searchsorted(self.tbins, timerange[1])
            dt_s = np.s_[start:end]
        else:
            dt = self.tbins[-1] - self.tbins[0]
            dt_s = np.s_[:]

        # define relevant rates: fluor, spont_emit, stim_emit, quench
        # depends on fluorescence that is detected
        v0fluor = self.nlevels == 3 or (self.nlevels == 4 and
            self.detcell['fluorwl'] == '308')
        v1fluor = self.nlevels == 4 and self.detcell['fluorwl'] == '282'
        v1v0fluor = self.nlevels == 4 and self.detcell['fluorwl'] == 'both'
        if v0fluor:
            fluorpop = self.abcpop[dt_s,2,:].sum(1)
            spont_emit = self.rates['Aca']
            fluor = spont_emit
            quench = self.rates['kqc']['tot']*self.detcell['Q']
            # stim_emit needs to account for scaling by being only from single
            # rotational level within "fluorpop" and being a function of time
            # (whether uvlaser is on or not)
            rot_factor = np.empty_like(self.tbins[dt_s])
            # use idx_zeros to avoid divide-by-zero warning
            idx_zeros = self.abcpop[dt_s,2,:].sum(1) == 0
            rot_factor[idx_zeros] = 0
            rot_factor[~idx_zeros] = (self.abcpop[dt_s,2,0][~idx_zeros]/
                self.abcpop[dt_s,2,:].sum(1)[~idx_zeros])
            stim_emit = (intensity(self.tbins[dt_s], self.uvlaser) *
                    self.hline['Bcb'] * rot_factor)
        elif v1fluor:
            fluorpop = self.abcpop[dt_s,3,:].sum(1)
            spont_emit = self.rates['Ada'] + self.rates['Adb']
            fluor = self.rates['Ada']
            quench = (self.rates['kqc']['tot'] * self.detcell['Q']
                + self.rates['kqd_vib'] * self.detcell['Q'])
            # stim_emit calcs
            rot_factor = np.empty_like(self.tbins[dt_s])
            idx_zeros = self.abcpop[dt_s,3,:].sum(1) == 0
            rot_factor[idx_zeros] = 0
            rot_factor[~idx_zeros] = (self.abcpop[dt_s,3,0][~idx_zeros]/
                self.abcpop[dt_s,3,:].sum(1)[~idx_zeros])
            stim_emit = (intensity(self.tbins[dt_s], self.uvlaser) *
                    self.hline['Bcb'] * rot_factor)
        elif v1v0fluor:
            # treat 'c' and 'd' as single blob
            fluorpop = self.abcpop[dt_s,2:,:].sum(1).sum(1)
            spont_emit = self.rates['Ada'] + self.rates['Aca']
            fluor = self.rates['Ada'] + self.rates['Aca']
            quench = self.rates['kqc']['tot'] * self.detcell['Q']
            # stim_emit calcs
            # here rot_factor scales by fraction within line in 'd', compared
            # to total fluorescing pop in both 'c' AND 'd'
            rot_factor = np.empty_like(self.tbins[dt_s])
            idx_zeros = self.abcpop[dt_s,3,:].sum(1).sum(1) == 0
            rot_factor[idx_zeros] = 0
            rot_factor[~idx_zeros] = (self.abcpop[dt_s,3,0][~idx_zeros]/
                self.abcpop[dt_s,2:,:].sum(1).sum(1)[~idx_zeros])
            stim_emit = (intensity(self.tbins[dt_s], self.uvlaser) *
                    self.hline['Bcb'] * rot_factor)
        qyield = fluor / (spont_emit + stim_emit + quench)
        fluorescence = fluorpop * qyield
        # finally, take average over interval and report on per-second basis
        avgfluorecencerate = fluorescence.mean() / dt
        return avgfluorescencerate

    def solveode(self):
        '''Integrate ode describing two-photon LIF.

        Use master equation (no Jacobian) and all relevant parameters.
        
        Define global parameters that are independent of HITRAN OH IR data
        within function: Additional OH parameters related to 'c' state and
        quenching, and laser parameters. Also set up parameters for solving
        and plotting ODE.

        Outputs:
        --------
        N : ndarray
        Relative population of 'a', 'b' (and 'c') states over integration time.
        Three-dimensional array: first dimension time, second dimension a/b/c
        state, third dimension subpopulations within state.
        
        Subpopulations defined, in order, as (1) bins excited individually by
        swept IR laser (one bin if IR laser sweep off), (2) population in line
        wings not reached by swept IR laser (one always empty bin if IR laser
        sweep off), (3) other half of lambda doublet for a/b PI states, (4)
        other rotational levels within same vibrational level.
        '''

        self.logger.info('solveode: integrating at {} torr, {} K, OH in cell, '
            '{:.2g} cm^-3'.format(self.detcell['press'],self.detcell['temp'],
                self.detcell['ohtot']))
        tl = self.odepar['inttime'] # total int time

        # set-up steps only required if IR laser is swept:
        if self.dosweep:
            self.logger.info('solveode: sweep mode: {}'.format(self.sweep.stype))
            self.makeAbs()
            
            # Align bins for IR laser and absorbance features for integration
            self.sweep.alignBins(self.abfeat)

            # avg_bintime calced for 'sin'. 'saw' twice as long.
            avg_bintime = self.sweep.tsweep\
                /(2*self.sweep.width/self.sweep.binwidth)
            dt = avg_bintime/self.sweep.avg_step_in_bin
            self.tbins = np.arange(0, tl+dt, dt)
            t_steps = np.size(self.tbins)

            # define local variables for convenience
            num_las_bins=self.sweep.las_bins.size
            # lambda doublet, other rot
            tsweep = self.sweep.tsweep
            stype = self.sweep.stype

            # Determine location of swept IR (a to b) laser by defining 1D array
            # self.sweepfunc: las_bins index for each point in tsweep.
            tindex=np.arange(np.size(self.tbins))
            tindexsweep=np.searchsorted(self.tbins,tsweep,side='right')-1
            if stype=='saw':
                self.sweepfunc=np.floor((tindex%tindexsweep)*(num_las_bins)\
                    /tindexsweep)
            elif stype=='sin':
                self.sweepfunc = np.round((num_las_bins-1)/2.\
                    *np.sin(2*np.pi/tindexsweep*tindex)+(num_las_bins-1)/2.)
            else:
                self.sweepfunc= np.empty(np.size(tindex))
                self.sweepfunc.fill(np.floor(num_las_bins/2))

        else: # single 'bin' excited by laser. Set up in __init__
            dt = self.odepar['dt'] # s
            self.tbins = np.arange(0, tl+dt, dt)
            t_steps = np.size(self.tbins)
            tindex=np.arange(t_steps)
            self.sweepfunc= np.zeros(np.size(tindex))

        self.logger.info('solveode: integrating {:.2g} s, '.format(tl)+
            'step size {:.2g} s'.format(dt))

        # set up ODE

        # Create initial state N0, all pop distributed in ground state
        self.N0 = np.zeros((self.nlevels,self.sweep.las_bins.size+3))
        if self.dosweep:
            self.N0[0,0:-3] = self.abfeat.intpop * self.rotfrac[0] \
            * self.detcell['ohtot'] / 2
            self.N0[0,-3] = (self.abfeat.pop.sum() - self.abfeat.intpop.sum()) \
                *self.rotfrac[0] * self.detcell['ohtot'] / 2 # pop outside laser sweep
        else:
            self.N0[0,0] = self.detcell['ohtot'] * self.rotfrac[0] / 2
            self.N0[0,-3] = 0 # no population within rot level isn't excited. 
        # other half of lambda doublet
        self.N0[0,-2] = self.detcell['ohtot'] * self.rotfrac[0] / 2
        self.N0[0,-1] = self.detcell['ohtot'] * (1-self.rotfrac[0]) # other rot

        # Create array to store output at each timestep, depending on keepN:
        # N stores a/b/c state pops in each bin over time.
        # abcpop stores a/b/c pops, tracks in or out rot/lambda of interest.
        if self.odepar['keepN']:
            self.N=np.empty((t_steps,self.nlevels,self.sweep.las_bins.size+3))
            self.N[0] = self.N0
        else:
            # TODO rename abcpop to something better since ab abc abcd possible
            self.abcpop=np.empty((t_steps,self.nlevels,2))
            self.abcpop[0]=np.array([self.N0[:,0:-2].sum(1),
                self.N0[:,-2:].sum(1)]).T

        # Initialize scipy.integrate.ode object, lsoda method
        r = ode(self.dN)
        # r.set_integrator('vode',nsteps=500,method='bdf')
        r.set_integrator('lsoda', with_jacobian=False,)
        r.set_initial_value(list(self.N0.ravel()), 0)

        self.logger.info('  %  |   time   |   bin   ')
        self.logger.info('--------------------------')

        # Solve ODE
        self.time_progress=0 # laspos looks at this to choose sweepfunc index.
        old_complete=0 # tracks integration progress for self.logger
        while r.successful() and r.t < tl-dt:
            # display progress
            complete = r.t/tl
            if floor(complete*100/10)!=floor(old_complete*100/10):
                self.logger.info(' {0:>3.0%} | {1:8.2g} | {2:7.0f} '
                    .format(complete,r.t,self.sweepfunc[self.time_progress]))
            old_complete = complete
            
            # integrate
            entry=int(round(r.t/dt))+1
            nextstep = r.integrate(r.t + dt)
            nextstepN = np.resize(nextstep,
                                  (self.nlevels,self.sweep.las_bins.size + 3))

            # save output
            if self.odepar['keepN'] == True:
                self.N[entry] = nextstepN
            else:
                self.abcpop[entry] = np.array([nextstepN[:,0:-2].sum(1),
                    nextstepN[:,-2:].sum(1)]).T

            self.time_progress+=1

        self.logger.info('solveode: done with integration')

    def laspos(self):
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
            self.logger.warning('laspos: voigt_pos out of range')
        return voigt_pos

    def dN(self, t, y):
        '''Construct differential equations to describe 2- or 3-state model.

        Parameters:
        -----------
        t : float
        Time
        y: ndarray
        1D-array describing the population in each bin in each energy level.
        Flattened version of multidimensional array `N`.

        Outputs
        -------
        result : ndarray
        1D-array describing dN in all 'a' states, then 'b', ...
        '''
        # ode method requires y passed in and out of dN to be one-dimensional.
        # For calculations within dN, reshape y back into 2D form of N
        y = y.reshape(self.nlevels,-1)

        # laser intensities accounting for pulsing
        Lir = intensity(t, self.irlaser)
        Luv = intensity(t, self.uvlaser)

        # Represent position of IR laser with Lir_sweep
        if self.dosweep:
            voigt_pos = self.laspos()
        else: # laser always at single bin representing entire line
            voigt_pos = 0
        Lir_sweep = np.zeros(self.sweep.las_bins.size + 3)
        Lir_sweep[voigt_pos] = Lir

        # calculate fdist and fdist_lambda, distribution *within* single
        # rotational level or lambda level that is relaxed to
        if self.odepar['redistequil']:
            # equilibrium distribution in ground state, as calced for N0
            fdist = (self.N0[0,:-1] / self.N0[0,:-1].sum())
            fdist_lambda = fdist[:-1] / fdist[:-1].sum()
            # for A-state, assume distribution same as N0 but lambda-doublet is
            # non-existent.
            fdist_A = np.append(fdist_lambda, 0)
            fdist_array = np.vstack((fdist,fdist,fdist_A,fdist_A))
        elif y[0,0:-1].sum() != 0:
            # instantaneous distribution in ground state
            fdist = (y[0,:-1] / y[0,:-1].sum())
            fdist_lambda = fdist[:-1] / fdist[:-1].sum()
            fdist_A = np.append(fdist_lambda,0)
            fdist_array = np.vstack((fdist,fdist,fdist_A,fdist_A))
        else:
            fdist_array = np.zeros((y.shape[0], y.shape[1] - 1))
        
        '''To implement: cleaner implementation of 'internal' processes

        # within dN
        if self.solveode['rotequil']:
            rotarray = self.rotprocesses(y, t)
        else:
            rotarray = np.zeros_like(y)
        if self.solveode['lambdaequil']:
            lambdaarray = self.lambdaprocesses(y, t)
        else:
            lambdaarray = np.zeros_like(y)
        internalarray = rotarray + lambdaarray
        ratearray = vibronicarray + internalarray
        return ratearray

        def lambdaprocesses(self, y, t):
            levellambdarates = []
            for level in self.system:
                if level.term == 'pi':
                    levelrate = makeinternalrate('lambda')
                elif level.term == 'sigma':
                    levelrate = np.zeros_like(level)
                levellambdarates.append(levelrate)
            return np.vstack(levellambdarates)

        def rotprocesses(self, y, t):
            levelrotrates = []
            for level in self.system:
                if level.term == 'pi':
                    levelrate = makeinternalrate('rot_pi')
                elif level.term == 'sigma':
                    levelrate = makeinternalrate('rot_sigma')
            return np.vstack(levelrotrates)                    

        def self.makeinternalrate(ratetype, yl, equildist):
            internalrate = np.zeros_like(yl)
            baserate = internal_dict[ratetype][0]
            ratecoeff = internal_dict[ratetype][1]
            startrng = getrange(internal_dict[ratetype][2])
            endrng = getrange(internal_dict[ratetype][3])
            individuatedrates = (- yl[startrng] * baserate + yl[rngout]
                * baserate_reverse * equildist)
            internalrate[startrng] = individuatedrates
            internalrate[endrng] = -individuatedrates.sum()
            return internalrate

internal_dict = {'rot_pi':[self.rates.rr,
                           'quencher',
                           'rot_level',
                           'rot_other'],
                 'rot_sigma':[self.rates.rr,
                              'quencher',
                              'lambda_half',
                              'rot_other'],
                 'lambda':[self.rates.lambda,
                           'quencher',
                           'lambda_half',
                           'lambda_other']}



'''
        # generate rates for each process
        # vibronic rates
        vibroniclist = self.vibronicprocesses(y, t)
        vibronicarray = np.sum(vibroniclist.values(), axis=0)

        # rotational equilibration
        rrin = self.rates['rrout'] * self.rotfrac / (1 - self.rotfrac)
        rrates = np.array([self.rates['rrout'], rrin]).T
        if self.odepar['rotequil']:
            rrvalues = np.vstack([internalrate(yl, r * self.detcell['Q'], dist,
                'rot') for yl,r,dist in zip(y, rrates, fdist_array)])
        else:
            rrvalues = np.zeros_like(y)

        # lambda equilibration
        lrin = self.rates['lrout'] # assume equal equilibrium pops
        lrates = np.array([self.rates['lrout'], lrin]).T
        if self.odepar['lambdaequil']:
            lrvalues = np.vstack([internalrate(yl, r * self.detcell['Q'],
                fdist_lambda, 'lambda') for yl,r in zip(y, lrates)])
        else:
            lrvalues = np.zeros_like(y)

        result = vibronicarray + rrvalues + lrvalues
        # flatten to 1D array as required
        return result.ravel()


    def ratecoeff(self,ratetype,t):
        '''Value to multiply base rate by to get first-order rate constant.
        
        Time-dependent when 'coeff' is pulsed laser intensity.
        '''
        if ratetype == 'ir_laser':
            coeff = intensity(t, self.irlaser)
        elif ratetype == 'uv_laser':
            coeff = intensity(t, self.uvlaser)
        elif ratetype == 'quencher':
            coeff = self.detcell['Q']
        else:
            coeff = 1
        return coeff

    def vibronicprocesses(self, y, t):
        '''Create dict of rates for vibronic processes at given instant.

        Builds dict from those processes in vibronic_dict that start and end in
        vibronic levels that are included in the given KineticsRun instance.
        '''
        vibronicratearrays = {}
        for process in vibronic_dict:
            if ((self.startlevel_idx(process) < self.nlevels) and 
                    (self.endlevel_idx(process) < self.nlevels)):
                ratearray = np.zeros_like(y)
                startlevel = self.startlevel_idx(process)
                startrng = self.getrngidx(getstartrng(process))()
                endlevel = self.endlevel_idx(process)
                endrng = self.getrngidx(getendrng(process))()

                base = self.baserate_k(process)
                coeff = self.ratecoeff(coefftype(process),t)
                conc = y[startlevel,startrng]
                rate = base*coeff*conc
                # for vibronic process, startrng = endrng
                ratearray[startlevel,startrng] = -rate
                ratearray[endlevel,endrng] = rate
                vibronicratearrays.update({process:ratearray})
        return vibronicratearrays

    def getrngidx(self, rnglabel):
        '''Look up slice for named range within level

        Returns an unbound function because las_bin location is function of
        time.
        '''
        rng = {'las_bin': self.laspos,
                'half_lambda': lambda: np.s_[:-2],
                'rot_level': lambda: np.s_[:-1],
                'rot_other': lambda: np.s_[-1],
                'full': lambda: np.s_[:]}
        return rng[rnglabel]

    def popsfigure(self, title='population dynamics', subpop=None):
        '''For solved KineticsRun, create figure plotting selected subpops.

        Requires `N` as created by `KineticsRun.solveode()`.

        Parameters
        ----------
        title : str
        Title to display at top of plot.

        yl : str
        Y-axis label to display.

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
        
        fig, (ax0) = plt.subplots()

        # default plot
        if subpop is None:
            subpop = ['blp']

        # make twinx if any 'c' or 'd' will be plotted
        maketwinx = any([(s[0] == 'c') or (s[0] == 'd') for s in subpop])
        if maketwinx:
            ax1 = ax0.twinx()
            fig.subplots_adjust(right=0.8)
            ax1.set_ylabel('sigma state populations (c and/or d)')

        for plotcode in subpop:
            if plotcode[0] == 'a' or plotcode[0] == 'b':
                ax0.plot(self.tbins*1e6, self.popseries(plotcode), label=plotcode)
                if maketwinx:
                    ax1._get_lines.color_cycle.next()
            elif plotcode[0] == 'c' or plotcode[0] == 'd':
                ax1.plot(self.tbins*1e6, self.popseries(plotcode), label=plotcode)
                ax0._get_lines.color_cycle.next()
            else:
                raise NameError("improper plotcode ", plotcode)

        ax0.set_title(title)
        ax0.set_xlabel('Time ($\mu$s)')
        ax0.set_ylabel('pi state pops (a or b)')

        # gather up lines and labels from both axes for unified legend
        lines_list = [ax.get_legend_handles_labels()[0] for ax in fig.get_axes()]
        lines = [items for sublists in lines_list for items in sublists]
        labels_list = [ax.get_legend_handles_labels()[1] for ax in fig.get_axes()]
        labels = [items for sublists in labels_list for items in sublists]

        topax = fig.get_axes()[-1]
        topax.legend(lines, labels)

        return fig

    def popseries(self, plotcode):
        '''calculate 1D array describing timeseries of subpopulation.
        '''
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
        # use existing slicedict for sublevel
        sublevelslice = slicedict[sublevelabbrevdict[plotcode[1]]]
        # all sublevels used here should be a slice, so num requires .sum(1).
        # however, just in case sublevel slice is single entry, check ndim of
        # numraw to see if .sum(1) is necessary.
        numraw = self.N[:, levelidx, sublevelslice]
        if numraw.ndim == 1:
            num = numraw
        else:
            num = numraw.sum(1)

        denomabbrev = plotcode[2]
        if denomabbrev == 'd':
            denom = self.N[:, levelidx, slicedict['rot_level']].sum(1)
        elif denomabbrev == 'l':
            denom = self.N[:, levelidx, :].sum(1)
        elif denomabbrev == 'p':
            denom = self.detcell['ohtot']
        elif denomabbrev == 'a':
            denom = 1
        else:
            raise NameError("improper plotcode ", plotcode)

        popseries = num/denom
        return popseries

    def vslaserfigure(self,func,title='plot',yl='y axis',pngout=None):
        '''Make arbitrary plot in time with laser sweep as second plot
        
        Parameters
        ----------
        func : ndarray
        1D set of values that is function of self.tbins

        title : str
        Title to display on top of plot

        yl : str
        Y-axis label to display.

        pngout : str
        filename to save PNG output. Displays plot if not given.
        '''
        if hasattr(self,'abcpop')==False and hasattr(self,'N')==False:
            self.logger.warning('need to run solveode first!')
            return
        elif hasattr(self,'abcpop')==False and self.odepar['keepN']:
            self.abcpop = np.empty((np.size(self.tbins),self.nlevels,2))
            self.abcpop[:,:,0]=self.N[:,:,0:-2].sum(2)
            self.abcpop[:,:,1]=self.N[:,:,-2:].sum(2)
        
        fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
        fig.subplots_adjust(hspace=.3)
        ax0.plot(self.tbins*1e6,func)
        ax0.set_title(title)
        ax0.set_ylabel(yl)

        time_indices=np.arange(np.size(self.tbins))
        if self.dosweep:
            ax1.plot(self.tbins*1e6,
                self.sweep.las_bins[self.sweepfunc[time_indices].astype(int)]/1e6)
        else:
            ax1.plot(self.tbins*1e6,self.tbins*0)
        ax1.set_title('Position of IR beam')
        ax1.set_xlabel('Time ($\mu$s)')
        ax1.set_ylabel('Relative Frequency (MHz)')

        return fig

    def absfigure(self,laslines=True):
        '''Plot the calculated absorption feature in frequency space
        
        Requires KineticsRun instance with an Abs that makeProfile has been run
        on (i.e., have self.abfeat.abs_freq and self.abfeat.pop)

        Parameters
        ----------
        laslines : Bool
        Whether to plot the edges of where the laser sweeps. Requires the
        KineticsRun instance to have a Sweep with self.sweep.las_bins array.
        '''
        fig, (ax0) = plt.subplots(nrows=1)
        ax0.plot(self.abfeat.abs_freq/1e6,self.abfeat.pop)
        ax0.set_title('Calculated absorption feature, ' \
            + str(self.detcell['press'])+' torr')
        ax0.set_xlabel('Relative frequency (MHz)')
        ax0.set_ylabel('Relative absorption')
        
        if laslines:
            ax0.axvline(self.sweep.las_bins[0]/1e6,ls='--')
            ax0.axvline(self.sweep.las_bins[-1]/1e6,ls='--')
        return fig

    def savecsv(self, csvout):
        '''save csv of 3-level system populations and time values

        first column is time, next three columns are populations of a, b and
        c in state of interest.'''

        if hasattr(self,'abcpop')==False and hasattr(self,'N')==True:
            self.abcpop = np.empty((np.size(self.tbins),self.nlevels,2))
            self.abcpop[:,:,0]=self.N[:,:,0:-2].sum(2)
            self.abcpop[:,:,1]=self.N[:,:,-2:].sum(2)
        timeseries = self.tbins[:, np.newaxis] # bulk out so ndim = 2
        abcpop_slice = self.abcpop[:,:,0] # slice along states of interest
        np.savetxt(csvout,np.hstack((timeseries,abcpop_slice)),
                delimiter = ",", fmt="%.6e")

    def saveOutput(self,file):
        '''Save result of solveode to npz file.
        
        Saves arrays describing the population over time, laser bins, tbins,
        sweepfunc, absorption frequencies, and Voigt profile.
        
        Parameters
        ----------
        file : str
        Path of file to save output (.npz extension standard).
        '''
        np.savez(file,
            abcpop=self.abcpop,
            las_bins=self.sweep.las_bins,
            tbins=self.tbins,
            sweepfunc=self.sweepfunc,
            abs_freq=self.abfeat.abs_freq,
            pop=self.abfeat.pop)

    def loadOutput(self,file):
        '''Populate KineticsRun instance with results saved to npz file.

        Writes to values for abcpop, sweep.las_bins, tbins, sweepfunc, abfeat,
        abfeat.abs_freq and abfeat.pop.

        Parameters
        ----------
        file : str
        Path of npz file with saved output.
        '''
        with np.load(file) as data:
            self.abcpop=data['abcpop']
            self.sweep.las_bins=data['las_bins']
            self.tbins=data['tbins']
            self.sweepfunc=data['sweepfunc']
            self.abfeat = Abs(0)
            self.abfeat.abs_freq=data['abs_freq']
            self.abfeat.pop=data['pop']

    # interpret ratetype parameters in terms of particular KineticsRun instance
    def baserate_k(self, ratetype):
        return getnested(baserate(ratetype), self.rates)
    def startlevel_idx(self, ratetype):
        return self.levelsidx[startlevel(ratetype)]
    def endlevel_idx(self, ratetype):
        return self.levelsidx[endlevel(ratetype)]

def getnested(keys, d):
    '''Access data in nested dict using a list of keys.'''
    if not isinstance(keys, list): # handle single key
        keys = [keys]
    return reduce(dict.__getitem__, keys, d)

###############################################################################
# GLOBAL VARS
levels = {'pi_v0': 0, 'pi_v1': 1, 'sigma_v0': 2, 'sigma_v1': 3}

slicedict = {'rot_level': np.s_[:-1],
             'rot_other': np.s_[-1],
             'lambda_half': np.s_[:-2],
             'lambda_other': np.s_[-2],
             'swept': np.s_[:-3],
             'half_lambda': np.s_[:-2],
             'rot_level': np.s_[:-1],
             'rot_other': np.s_[-1],
             'full': np.s_[:]}
# Form of vibronic_dict:
# rate constant, 'concentration' it needs to be multiplied by (or `None` if
# first-order), initial level, final level, initial range (within level), final
# range.
vibronic_dict = {'absorb_ir':['Bba',
                              'ir_laser',
                              'pi_v0',
                              'pi_v1',
                              'las_bin',
                              'las_bin'],
                 'absorb_uv':['Bcb',
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
                 'vib_quench_p1p0':[['kqb','tot'],
                                    'quencher',
                                    'pi_v1',
                                    'pi_v0',
                                    'full',
                                    'full'],
                 'elec_quench_s0p0':[['kqc','tot'],
                                     'quencher',
                                     'sigma_v0',
                                     'pi_v0',
                                     'full',
                                     'full'],
                 'elec_quench_s1p0':[['kqc','tot'],
                                     'quencher',
                                     'sigma_v1',
                                     'pi_v0',
                                     'full',
                                     'full'],
                 'spont_emit_s0p0':['Aca',
                                    None,
                                    'sigma_v0',
                                    'pi_v0',
                                    'full',
                                    'full'],
                 'spont_emit_s0p1':['Acb',
                                    None,
                                    'sigma_v0',
                                    'pi_v1',
                                    'full',
                                    'full'],
                 'spont_emit_s1p0':['Ada',
                                    None,
                                    'sigma_v1',
                                    'pi_v0',
                                    'full',
                                    'full'],
                 'spont_emit_s1p1':['Adb',
                                    None,
                                    'sigma_v1',
                                    'pi_v1',
                                    'full',
                                    'full'],
                 'vib_quench_s1s0':[['kqc','tot'],
                                    'quencher',
                                    'sigma_v1',
                                    'sigma_v0',
                                    'full',
                                    'full']}
# accessors for vibronic_dict
def baserate(ratetype):
    return vibronic_dict[ratetype][0]
def coefftype(ratetype):
    return vibronic_dict[ratetype][1]
def startlevel(ratetype):
    return vibronic_dict[ratetype][2]
def endlevel(ratetype):
    return vibronic_dict[ratetype][3]
def getstartrng(ratetype):
    return vibronic_dict[ratetype][4]
def getendrng(ratetype):
    return vibronic_dict[ratetype][5]

def buildsystem(nlevels):
    levela = {'term': 'pi', 'startof': ['absorb_ir']}
    levelb = {'term': 'pi', 'startof': ['absorb_uv',
                                        'vib_quench',
                                        'stim_emit_ir',
                                        'spont_emit']}
    levelc = {'term': 'sigma', 'startof' : ['elec_quench',
                                           'stim_emit_uv',
                                           'spont_emit']}
    leveld = {'term': 'sigma', 'startof' : ['elec_quench',
                                           'stim_emit_uv',
                                           'spont_emit',
                                           'vib_quench']}
    if nlevels == 2:
        return [levela, levelb]
    elif nlevels == 3:
        return [levela, levelb, levelc]
    else:
        return [levela, levelb, levelc, leveld]

def intensity(t, laser):
    '''Calculate spec intensity of laser at given array of times.

    Assumes total integration time less than rep rate.
    '''
    t = np.asarray(t)
    scalar_input = False
    if t.ndim == 0:
        t = t[None] # convert to 1D
        scalar_input = True
    L = np.zeros_like(t)
    area = np.pi*(laser['diam']*0.5)**2
    spec_intensity = oh.spec_intensity(laser['power'],area,laser['bandwidth'])
    if not(laser['pulse']):
        L.fill(spec_intensity)
    else:
        whenon = (t>laser['delay']) & (t<laser['pulse']+laser['delay'])
        L[whenon] = spec_intensity

    if scalar_input:
        return np.squeeze(L)
    return L

def vibronicrate(y, rateconst, start, final):
    '''calculate contribution to overall rate array for process that
    goes between a/b/c states.

    Parameters
    ----------
    y : array
    Array of populations.
    rateconst : float or array
    First-order rate constant for process, s^-1
    start : int (0, 1 or 2)
    Starting level for rate process (a/b/c)
    final: int (0, 1 or 2)
    Final level for rate process (a/b/c)

    Output
    ------
    term : np.ndarray
    2D array shaped like y containing rates for process
    '''
    term = np.zeros_like(y)
    rate = rateconst * y[start]
    term[start] = -rate
    term[final] = rate
    return term
    
def internalrate(yl, ratecon, equildist, ratetype):
    '''calculate contribution to overall rate array for process
    internal to single a/b/c level with equilibrium distribution.

    Parameters
    ----------
    yl : array
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
    term = np.zeros_like(yl)
    if ratetype == 'rot':
        rngin = np.s_[0:-1]
        rngout = np.s_[-1]
    elif ratetype == 'lambda':
        rngin = np.s_[0:-2]
        rngout = np.s_[-2]
    else:
        ValueError('ratetype needs to be \'rot\' or \'lambda\'')

    if yl[rngin].sum() != 0:
        term[rngin] = (-yl[rngin] * ratecon[0] +
                yl[rngout] * ratecon[1] * equildist)
        term[rngout] = yl[rngin].sum() * ratecon[0] - yl[rngout] * ratecon[1]
    else:
        term.fill(0)
    return term
