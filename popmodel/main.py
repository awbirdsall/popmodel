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
import ConfigParser
logging.basicConfig(level=logging.INFO)

class Sweep(object):
    '''
    Represent sweeping parameters of the laser. Before performing solveode on
    a KineticsRun, need to alignBins of Sweep: adjust Sweep parameters to
    match Abs and align bins of Abs and Sweep. solveode does this.
    '''
    def __init__(self,
        stype='sin',
        tsweep=1.e-4,
        width=500.e6,
        binwidth=1.e6,
        factor=.1,
        keepTsweep=False,
        keepwidth=False):

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
                logging.info('alignBins: IR sweep width maximized: {:.2g} MHz'
                    .format(self.width/1e6))
                abmid = floor(np.size(abfeat.abs_freq)/2.)
                irfw=self.width/self.binwidth
                self.las_bins=abfeat.abs_freq[abmid-irfw/2:abmid+irfw/2]
                abfeat.intpop=abfeat.pop[abmid-irfw/2:abmid+irfw/2]
            else: # reduce self.width to abswidth
                fullwidth=self.width
                if self.keepTsweep==False: # scale tsweep by width reduction
                    self.tsweep=self.tsweep*abswidth/fullwidth 
                    logging.info('alignBins: IR sweep time reduced to {:.2g} \
                        s'.format(self.tsweep))
                else:
                    logging.info('alignBins: IR sweep time maintained at \
                        {:.2g} s'.format(self.tsweep))
                self.width=abswidth
                self.las_bins = abfeat.abs_freq[start:end]
                abfeat.intpop=abfeat.pop[start:end] # integrated pop
                logging.info('alignBins: IR sweep width reduced to {:.2g} MHz'
                    .format(abswidth/1e6))

        else:
            # Keep initial width, but still align bins to abfeat.abs_freq
            logging.info('alignBins: maintaining manual width and tsweep')
            start = np.where(abfeat.abs_freq>=self.las_bins[0])[0][0]
            end = np.where(abfeat.abs_freq<=self.las_bins[-1])[0][-1]
            self.las_bins=abfeat.abs_freq[start:end]
            self.width=self.las_bins[-1]-self.las_bins[0]+self.binwidth
            abfeat.intpop=abfeat.pop[start:end] # integrated pop
            logging.info('alignBins: sweep width {:.2g} MHz, \
                sweep time {:.2g} s'.format(self.width/1e6, self.tsweep))
        
        # report how much of the b<--a feature is being swept over:
        self.part_swept=np.sum(abfeat.intpop)
        logging.info('alignBins: region swept by IR beam represents {:.1%} \
            of feature\'s total population'.format(self.part_swept))

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
        logging.info('makeProfile: made abs profile')
        logging.info('makeProfile: abs profile has FWHM = {:.2g} MHz'
            .format(fwhm/1e6))
        logging.info('makeProfile: total width of stored array = {:.2g} MHz'
            .format(abswidth/1e6))

        # return np.array([abs_freq, pop])

class KineticsRun(object):
    '''Full model of OH population kinetics: laser, feature and populations.
    
    Has single instance of Sweep, describing laser dithering, and of Abs,
    describing absorption feature. Sweep is made in __init__, while Abs is made
    after the HITRAN file is imported and the absorption feature selected.
    '''
    def __init__(self,
        press=oh.op_press,
        temp=oh.temp,
        xoh=oh.xoh,
        stype='sin',
        delay=100,
        keepN=False,
        tsweep=1.e-4,
        width=500.e6,
        binwidth=1.e6,
        factor=.1,
        keepTsweep=False,
        keepwidth=False,
        withoutUV=False,
        rotequil=True,
        redistequil=True,
        lumpsolve=False):
        # Sweep object
        self.sweep=Sweep(stype=stype,
                        tsweep=tsweep,
                        width=width,
                        binwidth=binwidth,
                        factor=factor,
                        keepTsweep=keepTsweep,
                        keepwidth=keepwidth)

        # Operation parameters
        self.press=press # torr
        self.temp=temp # K
        self.xoh=xoh # mixing ratio of OH
        self.ohtot = atm.press_to_numdens(press,temp)*xoh

        # ODE solution parameters
        self.delay = delay # s, artificial UV delay to allow b to spin up
        # default delay huge to effectively 'turn off' UV pulse for now...
        self.keepN = keepN # 'True' keeps full N array -- lots of memory.
        self.withoutUV = withoutUV # don't bother with UV if not needed
        self.rotequil = rotequil
        self.redistequil = redistequil
        self.lumpsolve = lumpsolve
        
    def addhitran(self,file,a):
        # TODO refactor so processing HITRAN file and saving a processed line
        # to self.hline are separate functions. That way, can pass in existing
        # hline for interactive session rather than figuring out how to shove
        # that existing hline into a KineticsRun...
        '''Process HITRAN file and save a single line to self.hline.

        Rest of the processed HITRAN file is not added to self. Nothing
        returned.

        Parameters
        ----------
        file : str
        Path to hitran file (e.g., "13_hit12.par")

        a : int
        Line of processed hitran array to choose.
        '''
        hpar = loadHITRAN.processHITRAN(file)
        self.hline = hpar[a] # single set of parameters from hpar
        logging.info('addhitran: using {} line at {:.4g} cm^-1'
            .format(self.hline['label'], self.hline['wnum_ab']))

    def makeAbs(self):
        '''Make an absorption profile using self.hline and experimental
        parameters.
        '''
        # Set up IR b<--a absorption profile
        self.abfeat = Abs(wnum=self.hline['wnum_ab'])
        self.abfeat.makeProfile(press=self.press,
                                T=self.temp,    
                                g_air=self.hline['g_air'])

    def integratefromfile(self,file,linenumber,intperiods,avg_step_in_bin):



    def solveode(self, file='13_hit12.par', a=24, intperiods=2.1,
            avg_step_in_bin=20.):
        # TODO: refactor so this really is only the solveode step. Will provide
        # more flexibility in wrapping together scripts, e.g., not needing to
        # reimport a single HITRAN line over and over and over...
        '''Integrate ode describing two-photon LIF.

        Use master equation (no Jacobian) and all relevant parameters.
        
        Define global parameters that are independent of HITRAN OH IR data
        within function: Additional OH parameters related to 'c' state and
        quenching, and laser parameters. Also set up parameters for solving
        and plotting ODE.

        Parameters:
        -----------
        file : str
        Input HITRAN file path (160-char format). Default assumes file is in
        current directory with default HITRAN OH filename
        
        a : int
        index of transition to use, within processed HITRAN data file. a=24
        gives Q1(1) line in '13_hit12.par' using what has been the default
        filter parameters.

        intperiods : float
        Number of periods of the Sweep to integrate over.

        avg_step_in_bin : float
        Average number of integration steps to spend in each frequency bin as
        laser sweeps over frequencies. Default of 20 is conservative, keeps in
        mind that time in each bin is variable when sweep is sinusoidal.

        Outputs:
        --------
        N : ndarray
        Relative population of 'a', 'b' (and 'c') states over integration time.    
        '''
        logging.info('solveode: integrating at {} torr, {} K, OH in cell \
            {:.2g} cm^-3'.format(self.press,self.temp,self.ohtot))
        logging.info('solveode: sweep mode: {}'.format(self.sweep.stype))
        # Set up IR b<--a absorption profile from HITRAN
        self.addhitran(file,a)
        self.makeAbs()
        
        # Align bins for IR laser and absorbance features for integration
        self.sweep.alignBins(self.abfeat)

        # set integration time based on IR sweep time and intperiods argument
        tl = self.sweep.tsweep*intperiods # total integration time
        # avg_bintime calced for 'sin'. 'saw' twice as long.
        avg_bintime = self.sweep.tsweep\
            /(2*self.sweep.width/self.sweep.binwidth)
        dt = avg_bintime/avg_step_in_bin
        # checked: avg 20 avg_step_in_bin agrees with 100
        self.tbins = np.arange(0, tl+dt, dt)
        t_steps = np.size(self.tbins)

        logging.info('solveode: integrating {:.2g} s, \
            step size {:.2g} s'.format(tl,dt))

        # define local variables for convenience
        num_las_bins=np.size(self.sweep.las_bins)
        num_int_bins=num_las_bins+2 # +2 = outside laser sweep, other rot
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

        # set up ODE
        self.time_progress=0 # laspos looks at this to choose sweepfunc index.

        # Create initial state N0, all pop distributed in ground state
        if self.withoutUV:
            self.nlevels=2
        else:
            self.nlevels=3    

        # assume for now dealing with N"=1.
        self.N0 = np.zeros((self.nlevels,num_int_bins))
        self.N0[0,0:-2] = self.abfeat.intpop * oh.rotfrac[0] * self.ohtot
        self.N0[0,-2] = (self.abfeat.pop.sum() - self.abfeat.intpop.sum()) \
            *oh.rotfrac[0] * self.ohtot # pop outside laser sweep
        self.N0[0,-1] = self.ohtot * (1-oh.rotfrac[0]) # other rot levels

        # Create array to store output at each timestep, depending on keepN
        # N stores a/b/c state pops in each bin over time
        # abcpop stores a/b/c pops, tracks in or out rot level of interest.
        if self.keepN:
            self.N=np.empty((t_steps,self.nlevels,num_int_bins))
            self.N[0] = self.N0
        else:
            self.abcpop=np.empty((t_steps,self.nlevels,2))
            self.abcpop[0]=np.array([self.N0[:,0:-1].sum(1),self.N0[:,-1]]).T

        # Initialize scipy.integrate.ode object, lsoda method
        r = ode(self.dN)
        r.set_integrator('lsoda', with_jacobian=False)
        if self.lumpsolve:
            self.N0lump=self.makeNlump(self.N0)
            r.set_initial_value(list(self.N0lump.ravel()), 0)
        else:
            r.set_initial_value(list(self.N0.ravel()), 0)

        logging.info('  %  |   time   |   bin   ')
        logging.info('--------------------------')

        # Solve ODE
        old_complete=0 # tracks integration progress for logging
        while r.successful() and r.t < tl-dt:
            # display progress
            complete = r.t/tl

            if floor(complete*100/10)!=floor(old_complete*100/10):
                logging.info(' {0:>3.0%} | {1:8.2g} | {2:7.0f} '
                    .format(complete,r.t,self.sweepfunc[self.time_progress]))
            old_complete = complete
            
            # integrate
            entry=int(round(r.t/dt))+1
            nextstep = r.integrate(r.t + dt)
            nextstepN = np.resize(nextstep, (self.nlevels,num_int_bins))

            # save output
            if self.keepN == True:
                self.N[entry] = nextstepN
            else:
                self.abcpop[entry] = np.array([nextstepN[:,0:-1].sum(1),
                    nextstepN[:,-1]]).T

            self.time_progress+=1

        logging.info('solveode: done with integration')

    def makeNlump(self,N):
        '''Try to make solveode more efficient by lumping together all N that
        the laser sweeps over, apart from the current bin. Developmental.

        Parameters
        ----------
        N : ndarray
        2D array as constructed in solveode that stores population across the
        absorption profile and in each energy level, at a single timestep.

        Outputs
        -------
        out : ndarray
        2D array that lumps together all population that the laser sweeps over,
        apart from the current laser bin, in the ground state.
        '''
        out=np.zeros((self.nlevels,4))
        out[0,0]=N[0,self.laspos()]
        out[0,1]=np.sum(N[0,:])-out[0,0]
        out[0,-2:]=N[0,-2:] # same values outside laser sweep, other rot
        # need to do anything with upper states? as written, they're all 0
        return out

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
        num_las_bins=np.size(self.sweep.las_bins)
        num_int_bins=num_las_bins+2
        if voigt_pos+1 > num_las_bins:
            logging.warning('laspos: voigt_pos out of range')
        return voigt_pos

    def dN(self, t, y):
        '''Construct differential equations to describe 2- or 3-state model.

        Parameters:
        -----------
        t : float
        Time
        y: ndarray
        1D-array describing the population in each bin in each energy level.
        ODE solver requires this to be a 1D-array, so pass in list(arr.ravel())
        where `arr` is the multidimensional array. Within dN, this
        multidimensional array is reconstructed and then flattened again at the
        end.

        Outputs
        -------
        result : ndarray
        1D-array describing dN in all 'a' states, then 'b', ...
        '''

        # Define parameters from OH literature
        Acb = oh.Acb
        Aca = oh.Aca
        kqb = oh.kqb
        kqc = oh.kqc

        # Define parameters inherent to laser operation
        Lab = oh.Lab
        Lbc = oh.Lbc
        period = oh.period_UV
        pulsewidth_UV = oh.pulsewidth_UV

        # Define parameters dependent on KineticsRun instance:
        Bab = self.hline['Bab']
        Bba = self.hline['Bba']
        Bbc = self.hline['Bbc']
        Bcb = self.hline['Bcb']
        Q = atm.press_to_numdens(self.press, self.temp) # quencher conc

        # Represent position of IR laser with Lab_sweep
        # smaller integration matrix with lumpsolve:
        if self.lumpsolve:
            Lab_sweep=np.array([Lab,0])

        else:
            voigt_pos=self.laspos()
            num_int_bins=np.size(self.sweep.las_bins)+2
            Lab_sweep=np.zeros(num_int_bins)
            Lab_sweep[voigt_pos]=Lab

        # reshape y back into form where each nested 1D array contains all
        # populations in given energy level:
        y=y.reshape(self.nlevels,-1)
        
        # Calculate coefficients for...
        # ...b<--a:
        absorb_ab = Bab * Lab_sweep * y[0]

        # ...b-->a: (spont emission negligible)
        stim_emit_ba = Bba * Lab_sweep * y[1]
        quench_b = kqb * Q * y[1]

        # ...rotational relaxation:
        # convention: positive = gain pop in rot state of interest
        rrout = np.array([7.72e-10,7.72e-10, 4.65e-10])
        # Values from Smith and Crosley, 1990. Undifferentiated for quencher
        # or vibrational state
        rrin = rrout * oh.rotfrac/(1-oh.rotfrac)

        if self.redistequil:
            fdist = (self.N0[0,0:-1]/self.N0[0,0:-1].sum())
        elif y[0,0:-1].sum() != 0:
            fdist = (y[0,0:-1]/y[0,0:-1].sum())
        else:
            fdist = 0

        # if UV laser calcs are off, only have a and b states:
        if self.withoutUV:
            dN0 = - absorb_ab + stim_emit_ba + quench_b
            dN1 = absorb_ab - stim_emit_ba - quench_b
            intermediate = np.array([dN0, dN1])

            rrvalues = np.empty_like(intermediate)
            if self.rotequil:
                rrvalues[0,0:-1] = -y[0,0:-1]*Q*rrout[0] \
                +y[0,-1]*Q*rrin[0]*fdist
                # assuming repopulation from other rotational levels flows to 
                # rot level of interest based on equilibrium distribution
                rrvalues[0,-1] = y[0,0:-1].sum()*Q*rrout[0] \
                -y[0,-1]*Q*rrin[0]
                
                if y[1,0:-1].sum() != 0: # avoid divide by zero error
                    rrvalues[1,0:-1] = -y[1,0:-1]*Q*rrout[1]\
                        +y[1,-1]*Q*rrin[1]*fdist
                    # approx equilibrium distribution same as v"=0
                    rrvalues[1,-1] = y[1,0:-1].sum()*Q*rrout[1] \
                        -y[1,-1]*Q*rrin[1]
                else:
                    rrvalues[1,:] = 0
            else:
                rrvalues.fill(0)
            result = (intermediate + rrvalues).ravel()

            return result

        # if UV laser calcs are on, a little more to do:
        else:
            # Pulse the UV (b to c) laser (assume total time < rep rate):
            if t>self.delay and t<pulsewidth_UV+self.delay:
                Lbc=Lbc
            else:
                Lbc = 0

            # c<--b processes:
            absorb_bc = Bbc * Lbc * y[1]

            # c-->a processes:
            spont_emit_ca = Aca * y[2]
            quench_c = kqc * Q * y[2]

            # c-->b processes:
            stim_emit_cb = Bcb * Lbc * y[2]
            spont_emit_cb = Acb * y[2]

            dN0 = - absorb_ab + stim_emit_ba + spont_emit_ca + quench_b \
                + quench_c
            dN1 = absorb_ab - stim_emit_ba + stim_emit_cb - absorb_bc \
                - quench_b + spont_emit_cb
            dN2 = - spont_emit_cb + absorb_bc - spont_emit_ca - quench_c \
                - stim_emit_cb

            intermediate = np.array([dN0, dN1, dN2])

            rrvalues = np.empty_like(intermediate)
            if self.rotequil:
                rrvalues[0,0:-1] = -y[0,0:-1]*Q*rrout[0] \
                    + y[0,-1]*Q*rrin[0]*fdist
                # assuming repopulation from other rotational levels flows to 
                # rot level of interest based on equilibrium distribution
                rrvalues[0,-1] = y[0,0:-1].sum()*Q*rrout[0] \
                    - y[0,-1]*Q*rrin[0]
                
                if y[1,0:-1].sum() != 0: # avoid divide by zero error
                    rrvalues[1,0:-1] = -y[1,0:-1]*Q*rrout[1] \
                        + y[1,-1]*Q*rrin[1]*fdist
                    # approx equilibrium distribution same as v"=0
                    rrvalues[1,-1] = y[1,0:-1].sum()*Q*rrout[1] \
                        - y[1,-1]*Q*rrin[1]
                else:
                    rrvalues[1,:] = 0

                if y[2,0:-1].sum() != 0: # avoid divide by zero error
                    rrvalues[2,0:-1] = -y[2,0:-1]*Q*rrout[2] \
                        + y[2,-1]*Q*rrin[2]*fdist
                    rrvalues[2,-1] = y[2,0:-1].sum()*Q*rrout[2] \
                        - y[2,-1]*Q*rrin[2]
                else:
                    rrvalues[2,:] = 0

            else:
                rrvalues.fill(0)

            result = (intermediate + rrvalues).ravel()
            # flatten to 1D array: 1st all 'a' states entries, then 'b', ...:
            return result

        
    def plotpops(self,
        title='Relative population in vibrationally excited state',
        yl='Fraction of total OH'):
        '''Given solution N to solveode, plot 'b' state population over time.

        Requires:
        -either 'abcpop' or 'N' (to make 'abcpop') from solveode input
        -to make 'abcpop' from 'N', need tbins, nlevels

        Parameters
        ----------
        title : str
        Title to display at top of plot.

        yl : str
        Y-axis label to display.
        '''
        if hasattr(self,'abcpop')==False and hasattr(self,'N')==False:
            logging.warning('need to run solveode first!')
            return
        elif hasattr(self,'abcpop')==False and hasattr(self,'N')==True:
            self.abcpop = np.empty((np.size(self.tbins),self.nlevels,2))
            self.abcpop[:,:,0]=self.N[:,:,0:-1].sum(2)
            self.abcpop[:,:,1]=self.N[:,:,-1]

        self.plotvslaser(self.abcpop[:,1,0]/self.ohtot,title,yl)

    def plotvslaser(self,func,title='plot',yl='y axis'):
        '''Make arbitrary plot in time with laser sweep as second plot
        
        Parameters
        ----------
        func : ndarray
        1D set of values that is function of self.tbins

        title : str
        Title to display on top of plot

        yl : str
        Y-axis label to display.
        '''
        if hasattr(self,'abcpop')==False and hasattr(self,'N')==False:
            logging.warning('need to run solveode first!')
            return
        elif hasattr(self,'abcpop')==False and self.keepN:
            self.abcpop = np.empty((np.size(self.tbins),self.nlevels,2))
            self.abcpop[:,:,0]=self.N[:,:,0:-1].sum(2)
            self.abcpop[:,:,1]=self.N[:,:,-1]
        
        fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
        fig.subplots_adjust(hspace=.3)
        ax0.plot(self.tbins*1e6,func)
        ax0.set_title(title)
        ax0.set_ylabel(yl)

        time_indices=np.arange(np.size(self.tbins))
        ax1.plot(self.tbins*1e6,
            self.sweep.las_bins[self.sweepfunc[time_indices].astype(int)]/1e6)
        ax1.set_title('Position of IR beam')
        ax1.set_xlabel('Time ($\mu$s)')
        ax1.set_ylabel('Relative Frequency (MHz)')
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
            + str(self.press)+' torr')
        ax0.set_xlabel('Relative frequency (MHz)')
        ax0.set_ylabel('Relative absorption')
        
        if laslines:
            ax0.axvline(self.sweep.las_bins[0],ls='--')
            ax0.axvline(self.sweep.las_bins[-1],ls='--')
        plt.show()

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

# Simple batch scripts

def pressdepen(file):
    '''Run solveode over range of pressures.

    Default solveode run, all output just printed with logging.info.

    Parameters
    ----------
    file : str
    Path to HITRAN file containing data.
    '''
    i=1
    pressconsidered=(2,10,100,760)
    for press in pressconsidered:
        logging.info('--------------------')
        logging.info('KineticsRun {:} OF {}'.format(i,
            np.size(pressconsidered)))
        logging.info('--------------------')
        k=KineticsRun(press=press,stype='sin')
        k.solveode(file)
        # k.plotpops()
        #k.abfeat=Abs()
        #k.abfeat.makeProfile(press=press)
        #k.sweep.matchAbsSize(k.abfeat)
        i+=1

def sweepdepen(file):
    '''Run solveode over range of sweep widths.

    Default solveode run, all output just printed with logging.info.

    Parameters
    ----------
    file : str
    Path to HITRAN file containing data.
    '''
    for factor in (0.01, 0.1, 0.5, 0.9):
        k=KineticsRun(stype='sin')
        k.sweep.factor=factor
        # k.abfeat=Abs()
        # k.abfeat.makeProfile()
        # k.sweep.matchAbsSize(k.abfeat)
        k.solveode(file)
        # k.plotpops()
