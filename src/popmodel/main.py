# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 10:49:58 2014

@author: Adam Birdsall

Kinetic model for two-photon OH LIF with focus on examining IR transition.

Capabilities:

- model population distribution across frequency space for v"=1 <-- v"=0
- model different options for sweeping IR laser freq over time
- use loadHITRAN to extract parameters from HITRAN file
- collect other physical and experimental parameters from ohcalcs
- integrate ODE describing population in quantum states
- consider populations both within and without rotational level of interest.
- turn off UV laser calculations an option to save memory

"""

# modules within package
import ohcalcs as oh
import atmcalcs as atm
import loadHITRAN as loadHITRAN

# other modules
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from scipy.constants import k as kb
from scipy.constants import c, N_A, pi
from scipy.integrate import ode
from math import floor
import logging
import argparse
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader # a lot slower sez https://stackoverflow.com/questions/18404441/why-is-pyyaml-spending-so-much-time-in-just-parsing-a-yaml-file

##############################################################################
# set up logging, follow python logging cookbook
# need to initialize here AND in each class/submodule
logger = logging.getLogger('popmodel')
logger.setLevel(logging.WARNING)
def stream_logging_info():
    logger.setLevel(logging.INFO)
    # console handler always runs (optional logfile through init_logfile())
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def init_logfile(logfile):
    '''set up FileHandler within logging to write to output file
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

def automate(hitfile,parameters,logfile=None,csvout=None,image=None,verbose=False):
    '''command-line-mode-style inputs to integration output
    '''
    if verbose:
        stream_logging_info()
    if logfile:
        init_logfile(logfile)
    # record to log each output filename
    argdict = {"log file":logfile,"output csv":csvout,
            "output png image":image}
    for (k,v) in argdict.iteritems():
        if v:
            logger.info('saving '+k+' to '+v)

    par = importyaml(parameters)
    hpar = loadHITRAN.processHITRAN(hitfile)
    k = KineticsRun(hpar,**par)
    k.solveode()
    if csvout:
        k.savecsv(csvout)
    if image:
        k.plotpops(pngout = image)

##############################################################################
class Sweep(object):
    '''
    Represent sweeping parameters of the laser. Before performing solveode on
    a KineticsRun, need to alignBins of Sweep: adjust Sweep parameters to
    match Abs and align bins of Abs and Sweep. runmodel does this.
    '''
    def __init__(self,
        stype='sin',
        tsweep=1.e-4,
        width=500.e6,
        binwidth=1.e6,
        factor=.1,
        keepTsweep=False,
        keepwidth=False):

        self.logger = logging.getLogger('popmodel.sweep')

        # parameters that don't change after initiated
        self.ircen=0 # set center of swept ir
        self.stype=stype # allowed: 'saw' or 'sin'. Anything else forces laser
        # to just sit at middle bin. Have used stype='None' for some calcs,
        # didn't bother to turn off alignBins -- some meaningless variables.
        self.binwidth=binwidth # Hz, have been using 1 MHz

        # initial sweep width and time -- alignBins can later reduce
        self.width=width # Hz
        self.tsweep=tsweep # s
        # max is 500 MHz for OPO cavity sweep, 100 GHz for seed sweep

        # make initial las_bins array
        self.makebins()

        # absorption cutoff relative to peak, used to reduce sweep width
        self.factor=factor

        # Whether alignbins readjusts sweep time or width
        self.keepTsweep=keepTsweep
        self.keepwidth=keepwidth

    def makebins(self):
        '''
        Used in self.__init__ to make initial las_bins array.
        '''
        self.las_bins = np.arange(self.ircen-self.width/2,
            self.ircen+self.width/2+self.binwidth,self.binwidth)

    def alignBins(self, abfeat):
        '''
        Adjust width, tsweep and las_bins of Sweep to match given abfeat Abs
        object.

        Parameters
        ----------
        abfeat : popmodel.application.Abs
        Absorption feature Sweep is aligned to. Must have abs_freq and pop
        (i.e., from Abs.makeProfile()).
        '''
        if self.keepwidth==False:
            # use absorption feature and cutoff factor to determine sweep size
            threshold = self.factor * np.max(abfeat.pop)
            abovecutoff = np.where(abfeat.pop > threshold)
            # start and end are indices defining the range of abfeat.abs_freq
            # to sweep over. abswidth is the size of the frequency range from
            # start to end.

            # use if...else to handle nonempty/empty abovecutoff result
            if np.size(abovecutoff) > 0:
                start =  abovecutoff[0][0]
                end = abovecutoff[0][-1]
                abswidth = abfeat.abs_freq[end]-abfeat.abs_freq[start]
            else:
                start = np.argmax(abfeat.pop)
                end = start + 1
                abswidth = self.binwidth

            # Default self.width defined in __init__ represents a physical cap
            # to the allowed dithering range. Only reduce self.width if the
            # frequency width obtained using the cutoff is less than that:
            if abswidth > self.width: # keep self.width maximized
                logger.info('alignBins: IR sweep width maximized: {:.2g} MHz'
                    .format(self.width/1e6))
                abmid = floor(np.size(abfeat.abs_freq)/2.)
                irfw=self.width/self.binwidth
                self.las_bins=abfeat.abs_freq[abmid-irfw/2:abmid+irfw/2]
                abfeat.intpop=abfeat.pop[abmid-irfw/2:abmid+irfw/2]
            else: # reduce self.width to abswidth
                fullwidth=self.width
                if self.keepTsweep==False: # scale tsweep by width reduction
                    self.tsweep=self.tsweep*abswidth/fullwidth 
                    logger.info('alignBins: IR sweep time reduced to '+
                        '{:.2g} s'.format(self.tsweep))
                else:
                    logger.info('alignBins: IR sweep time maintained at ,'
                        '{:.2g} s'.format(self.tsweep))
                self.width=abswidth
                self.las_bins = abfeat.abs_freq[start:end]
                abfeat.intpop=abfeat.pop[start:end] # integrated pop
                logger.info('alignBins: IR sweep width reduced to {:.2g} MHz'
                    .format(abswidth/1e6))

        else:
            # Keep initial width, but still align bins to abfeat.abs_freq
            logger.info('alignBins: maintaining manual width and tsweep')
            start = np.where(abfeat.abs_freq>=self.las_bins[0])[0][0]
            end = np.where(abfeat.abs_freq<=self.las_bins[-1])[0][-1]
            self.las_bins=abfeat.abs_freq[start:end]
            self.width=self.las_bins[-1]-self.las_bins[0]+self.binwidth
            abfeat.intpop=abfeat.pop[start:end] # integrated pop
            logger.info('alignBins: sweep width ',
                '{:.2g} MHz, sweep time {:.2g} s'.format(self.width/1e6,
                    self.tsweep))
        # report how much of the b<--a feature is being swept over:
        self.part_swept=np.sum(abfeat.intpop)
        logger.info('alignBins: region swept by IR beam represents '+
            '{:.1%} of feature\'s total population'.format(self.part_swept))

class Abs(object):
    '''absorbance line profile, initially defined in __init__ by a center 
    wavenumber `wnum` and a `binwidth`. Calling self.makeProfile then generates
    two 1D arrays:

    abs_freq : bins of frequencies (Hz)
    pop : relative population absorbing in each frequency bin

    pop is generated from abs_freq and the Voigt profile maker ohcalcs.voigt,
    which requires parameters that are passed through as makeProfile arguments
    (default are static parameters in ohcalcs). The formation of the two arrays
    is iterative, widening the abs_freq range by 50% until the edges of the pop
    array have less than 1% of the center.
    '''
    def __init__(self,wnum,binwidth=1.e6):
        self.logger = logging.getLogger('popmodel.Abs')
        self.wnum=wnum # cm^-1
        self.freq=wnum*c*100 # Hz
        self.binwidth=binwidth # Hz

    def __str__(self):
        return 'Absorbance feature centered at '+str(self.wnum)+' cm^-1'
      
    def makeProfile(self,abswidth=1000.e6,press=oh.op_press,T=oh.temp,
        g_air=oh.g_air,mass=oh.mass):
        '''
        Use oh.voigt func to create IR profile as self.abs_freq and self.pop.
    
        Parameters:
        -----------
        abswidth : float
        Minimum width of profile, Hz. Starting value that then expands if this
        does not capture 'enough' of the profile (defined as <1% of peak height
        at edges).

        press : float
        Operating pressure, torr. Defaults to ohcalcs value
    
        T : float
        Temperature. Defaults to ohcalcs value

        g_air : float
        Air-broadening coefficient provided in HITRAN files, cm^-1 atm^-1.
        Defaults to ohcalcs value.

        mass : float
        Mass of molecule of interest, kg. Defaults to ohcalcs value
        '''
        sigma=(kb*T / (mass*c**2))**(0.5)*self.freq # Gaussian std dev
    
        gamma=(g_air*c*100) * press/760. # Lorentzian parameter
        # air-broadened HWHM at 296K, HITRAN (converted from cm^-1 atm^-1)
        # More correctly, correct for temperature -- see Dorn et al. Eq 17

        # Make abs_freq profile, checking pop at edge <1% of peak
        enoughWidth=False 
        while enoughWidth==False:
            abs_freq = np.arange(-abswidth/2,
                abswidth/2+self.binwidth,
                self.binwidth)
            raw_pop=oh.voigt(abs_freq,1,0,sigma,gamma,True)
            norm_factor = 1/np.sum(raw_pop)
            pop=raw_pop * norm_factor # makes sum of pops = 1.
            if pop[0]>=0.01*np.max(pop):
                abswidth=abswidth*1.5
            else:
                enoughWidth=True
        self.abs_freq = abs_freq
        self.pop = pop
        startfwhm=np.where(pop>=np.max(pop)*0.5)[0][0]
        endfwhm=np.where(pop>=np.max(pop)*0.5)[0][-1]
        fwhm=abs_freq[endfwhm]-abs_freq[startfwhm]
        logger.info('makeProfile: made abs profile')
        logger.info('makeProfile: abs profile has FWHM = {:.2g} MHz'
            .format(fwhm/1e6))
        logger.info('makeProfile: total width of stored array = {:.2g} MHz'
            .format(abswidth/1e6))

        # return np.array([abs_freq, pop])

class KineticsRun(object):
    '''Full model of OH population kinetics: laser, feature and populations.
    
    If IR laser is swept, has single instance of Sweep, describing laser
    dithering, and of Abs, describing absorption feature. Sweep is made in
    __init__, while Abs is made after the HITRAN file is imported and the
    absorption feature selected.
    '''
    def __init__(self,hpar, irlaser,sweep,uvlaser,odepar,irline,uvline,detcell,rates):
        '''Initizalize KineticsRuns using dictionaries of input parameters and
        processed HITRAN file.
        
        The input parameters can be gathered up in a yaml file (in format of
        parameters.yaml) and passed in from the command line.
        '''
        self.logger = logging.getLogger('popmodel.KineticsRun')

        self.detcell = detcell
        self.detcell['ohtot'] = atm.press_to_numdens(detcell['press'],
                detcell['temp'])*detcell['xoh']
        # quencher conc
        self.detcell['Q'] = atm.press_to_numdens(self.detcell['press'], self.detcell['temp'])
        self.irlaser = irlaser
        self.uvlaser = uvlaser
        self.odepar = odepar
        self.irline = irline

        # Sweep object
        if sweep['dosweep']:
            self.dosweep=True
            self.sweep=Sweep(stype=sweep['stype'],
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
        logger.info('chooseline: using {} line at {:.4g} cm^-1'
            .format(self.hline['label'], self.hline['wnum_ab']))

        # extract rotation fraction for a b and c
        # figure out which 'F1' or 'F2' series that a and b state are:
        f_a = int(self.hline['label'][2]) - 1
        if self.hline['label'][3] == '(':
            f_b = f_a
        else:
            f_b = int(self.hline['label'][3]) - 1
        self.rotfrac = np.array([oh.rotfrac['a'][f_a][self.hline['Na']-1],
                                 oh.rotfrac['b'][f_b][self.hline['Nb']-1],
                                 oh.rotfrac['c'][self.hline['Nc']]])

    def makeAbs(self):
        '''Make an absorption profile using self.hline and experimental
        parameters.
        '''
        # Set up IR b<--a absorption profile
        self.abfeat = Abs(wnum=self.hline['wnum_ab'])
        self.abfeat.makeProfile(press=self.detcell['press'],
                                T=self.detcell['temp'],    
                                g_air=self.hline['g_air'])

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

        logger.info('solveode: integrating at {} torr, {} K, OH in cell, '
            '{:.2g} cm^-3'.format(self.detcell['press'],self.detcell['temp'],
                self.detcell['ohtot']))
        tl = self.odepar['inttime'] # total int time

        # set-up steps only required if IR laser is swept:
        if self.dosweep:
            logger.info('solveode: sweep mode: {}'.format(self.sweep.stype))
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
                    *np.sin(2*pi/tindexsweep*tindex)+(num_las_bins-1)/2.)
            else:
                self.sweepfunc= np.empty(np.size(tindex))
                self.sweepfunc.fill(np.floor(num_las_bins/2))

        else: # single 'bin' excited by laser. Set up in __init__
            dt = self.odepar['dt'] # s
            self.tbins = np.arange(0, tl+dt, dt)
            t_steps = np.size(self.tbins)
            tindex=np.arange(t_steps)
            self.sweepfunc= np.zeros(np.size(tindex))

        logger.info('solveode: integrating {:.2g} s, '.format(tl)+
            'step size {:.2g} s'.format(dt))

        # set up ODE
        if self.odepar['withoutUV']:
            self.nlevels=2
        else:
            self.nlevels=3    

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
        self.N0[0,-2] = self.detcell['ohtot'] * self.rotfrac[0] / 2 # other half of lambda doublet
        self.N0[0,-1] = self.detcell['ohtot'] * (1-self.rotfrac[0]) # other rot

        # Create array to store output at each timestep, depending on keepN:
        # N stores a/b/c state pops in each bin over time.
        # abcpop stores a/b/c pops, tracks in or out rot/lambda of interest.
        if self.odepar['keepN']:
            self.N=np.empty((t_steps,self.nlevels,self.sweep.las_bins.size+3))
            self.N[0] = self.N0
        else:
            self.abcpop=np.empty((t_steps,self.nlevels,2))
            self.abcpop[0]=np.array([self.N0[:,0:-2].sum(1),
                self.N0[:,-2:].sum(1)]).T

        # Initialize scipy.integrate.ode object, lsoda method
        r = ode(self.dN)
        # r.set_integrator('vode',nsteps=500,method='bdf')
        r.set_integrator('lsoda', with_jacobian=False,)
        r.set_initial_value(list(self.N0.ravel()), 0)

        logger.info('  %  |   time   |   bin   ')
        logger.info('--------------------------')

        # Solve ODE
        self.time_progress=0 # laspos looks at this to choose sweepfunc index.
        old_complete=0 # tracks integration progress for logger
        while r.successful() and r.t < tl-dt:
            # display progress
            complete = r.t/tl
            if floor(complete*100/10)!=floor(old_complete*100/10):
                logger.info(' {0:>3.0%} | {1:8.2g} | {2:7.0f} '
                    .format(complete,r.t,self.sweepfunc[self.time_progress]))
            old_complete = complete
            
            # integrate
            entry=int(round(r.t/dt))+1
            nextstep = r.integrate(r.t + dt)
            nextstepN = np.resize(nextstep, (self.nlevels,self.sweep.las_bins.size + 3))

            # save output
            if self.odepar['keepN'] == True:
                self.N[entry] = nextstepN
            else:
                self.abcpop[entry] = np.array([nextstepN[:,0:-2].sum(1),
                    nextstepN[:,-2:].sum(1)]).T

            self.time_progress+=1

        logger.info('solveode: done with integration')

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
            logger.warning('laspos: voigt_pos out of range')
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
        y=y.reshape(self.nlevels,-1)

        # laser intensities accounting for pulsing
        Lab = intensity(t, self.irlaser)
        Lbc = intensity(t, self.uvlaser)

        # Represent position of IR laser with Lab_sweep
        if self.dosweep:
            voigt_pos=self.laspos()
        else: # laser always at single bin representing entire line
            voigt_pos = 0
        Lab_sweep=np.zeros(self.sweep.las_bins.size + 3)
        Lab_sweep[voigt_pos]=Lab

        # calculate fdist and fdist_lambda, rotational and lambda distributions
        # that RET and lambda relaxation relax to.
        if self.odepar['redistequil']:
            # equilibrium distribution in ground state, as calced for N0
            fdist = (self.N0[0,0:-1]/self.N0[0,0:-1].sum())
            fdist_lambda = fdist[:-1]/fdist[:-1].sum()
        elif y[0,0:-1].sum() != 0:
            # instantaneous distribution in ground state
            fdist = (y[0,0:-1]/y[0,0:-1].sum())
            fdist_lambda = fdist[:-1]/fdist[:-1].sum()
        else:
            fdist = 0
        
        # generate rates for each process
        # rates between a/b/c states
        absorb_ab = abcrate(y, self.hline['Bab']*Lab_sweep,0,1)
        stim_emit_ba = abcrate(y, self.hline['Bba']*Lab_sweep,1,0)
        quench_b = abcrate(y, self.rates['kqb']['tot']*self.detcell['Q'],1,0)
        # logger.info(absorb_ab)
        intermediate = absorb_ab+stim_emit_ba+quench_b
        if not(self.odepar['withoutUV']):
            # Lbc only excites population in particular rot/lambda level
            Lbc_vec = np.zeros_like(y[0])
            Lbc_vec[0:-2].fill(Lbc)
            absorb_bc = abcrate(y, self.hline['Bbc']*Lbc_vec,1,2)
            spont_emit_ca = abcrate(y, self.rates['Aca'],2,0)
            quench_c=abcrate(y, self.rates['kqc']['tot']*self.detcell['Q'],2,0)
            stim_emit_cb = abcrate(y, self.hline['Bcb']*Lbc_vec,2,1)
            spont_emit_cb = abcrate(y, self.rates['Acb'],2,1)
            intermediate = (intermediate + absorb_bc + spont_emit_ca + quench_c
                + stim_emit_cb + spont_emit_cb)

        # rotational equilibration
        rrin = self.rates['rrout'] * self.rotfrac/(1-self.rotfrac)
        rrates = np.array([self.rates['rrout'],rrin]).T
        if self.odepar['rotequil']:
            rrvalues = np.vstack([internalrate(l, r, fdist, 'rot') for r,l
                in zip(rrates, y)])
        else:
            rrvalues = np.zeros_like(y)

        # lambda equilibration
        lrin = self.rates['lrout'] # assume equal equilibrium pops
        lrates = np.array([self.rates['lrout'],lrin]).T
        if self.odepar['lambdaequil']:
            lrvalues = np.vstack([internalrate(l, r, fdist_lambda, 'lambda')
                for r,l in zip(lrates, y)])
        else:
            lrvalues = np.zeros_like(y)

        result = intermediate + rrvalues + lrvalues
        # flatten to 1D array as required
        return result.ravel()


    def plotpops(self, title='excited state population', yl='b state pop',
            pngout = None):
        '''For solved KineticsRun, plot excited state population over time.

        Requires:
        -either 'abcpop' or 'N' (to make 'abcpop') from solveode input
        -to make 'abcpop' from 'N', need tbins, nlevels

        Parameters
        ----------
        title : str
        Title to display at top of plot.

        yl : str
        Y-axis label to display.

        pngout : str
        filename to save PNG output. Displays plot if not given.
        '''
        # make abcpop array if not already calculated
        if hasattr(self,'abcpop')==False and hasattr(self,'N')==False:
            logger.warning('need to run solveode first!')
            return
        elif hasattr(self,'abcpop')==False and hasattr(self,'N')==True:
            self.abcpop = np.empty((np.size(self.tbins),self.nlevels,2))
            self.abcpop[:,:,0]=self.N[:,:,0:-2].sum(2)
            self.abcpop[:,:,1]=self.N[:,:,-2:].sum(2)
        
        fig, (ax0) = plt.subplots()
        ax0.plot(self.tbins*1e6, self.abcpop[:,1,0]/self.detcell['ohtot'],
                'b-',label='b state pop')

        if self.nlevels == 3:
            ax1 = ax0.twinx()
            ax1.plot(self.tbins*1e6,
                    self.abcpop[:,2,0]/self.detcell['ohtot'], 'r-')
            ax1.set_ylabel('c state pop')
            ax0.plot(0,0,'r',label='c state pop') # dummy line for legend
            fig.subplots_adjust(right=0.9)

        ax0.set_title(title)
        ax0.set_ylabel(yl)
        ax0.legend()
        if pngout:
            fig.savefig(pngout)
        else:
            plt.show()    

    def plotvslaser(self,func,title='plot',yl='y axis',pngout=None):
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
            logger.warning('need to run solveode first!')
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
        if pngout:
            plt.savefig(pngout)
        else:
            plt.show()    

    def plotfeature(self,laslines=True):
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
            ax0.axvline(self.sweep.las_bins[0],ls='--')
            ax0.axvline(self.sweep.las_bins[-1],ls='--')
        plt.show()

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

def intensity(t, laser):
    '''Calculate spec intensity of laser at given time.

    Assumes total integration time less than rep rate.
    '''
    if not(laser['pulse']) or (t>laser['delay'] and
            t<laser['pulse']+laser['delay']):
        area = np.pi*(laser['diam']*0.5)**2
        L = oh.spec_intensity(laser['power'],area,laser['bandwidth'])
    else:
        L = 0
    return L

def abcrate(y, rateconst, start, final):
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
    1D array of population in single a/b/c level
    ratecon : list
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
    term = np.empty_like(yl)
    if ratetype == 'rot':
        rngin = np.s_[0:-1]
        rngout = np.s_[-1]
    elif ratetype == 'lambda':
        rngin = np.s_[0:-2]
        rngout = np.s_[-1]
    else:
        ValueError('ratetype needs to be \'rot\' or \'lambda\'')

    if yl[rngin].sum() != 0:
        term[rngin] = -yl[rngin]*ratecon[0] + yl[rngout]*ratecon[1]*equildist
        term[rngout] = yl[rngin].sum()*ratecon[0] - yl[rngout]*ratecon[1]
    else:
        term.fill(0)
    return term

##############################################################################
# Simple batch scripts, now deprecated because KineticsRun no longer assumes
# a bunch of default parameters

# def pressdepen(file):
#     '''Run solveode over range of pressures.

#     Default solveode run, all output just printed with logger.info.

#     Parameters
#     ----------
#     file : str
#     Path to HITRAN file containing data.
#     '''
#     i=1
#     pressconsidered=(2,10,100,760)
#     for press in pressconsidered:
#         logger.info('--------------------')
#         logger.info('KineticsRun {:} OF {}'.format(i,
#             np.size(pressconsidered)))
#         logger.info('--------------------')
#         k=KineticsRun(press=press,stype='sin')
#         k.solveode(file)
#         # k.plotpops()
#         #k.abfeat=Abs()
#         #k.abfeat.makeProfile(press=press)
#         #k.sweep.matchAbsSize(k.abfeat)
#         i+=1

# def sweepdepen(file):
#     '''Run solveode over range of sweep widths.

#     Default solveode run, all output just printed with logger.info.

#     Parameters
#     ----------
#     file : str
#     Path to HITRAN file containing data.
#     '''
#     for factor in (0.01, 0.1, 0.5, 0.9):
#         k=KineticsRun(stype='sin')
#         k.sweep.factor=factor
#         # k.abfeat=Abs()
#         # k.abfeat.makeProfile()
#         # k.sweep.matchAbsSize(k.abfeat)
#         k.solveode(file)
#         # k.plotpops()

##############################################################################
