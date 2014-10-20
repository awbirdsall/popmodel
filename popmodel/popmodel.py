# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 10:49:58 2014

@author: abirdsall
"""

"""
Kinetic model for two-photon OH LIF with focus on examining IR transition.

Capabilities:
--Model population distribution across frequency space for v"=1 <-- v"=0
transition.
--Model different options for sweeping IR laser freq over time (sawtooth, sin)
--Call processHITRAN to extract parameters from HITRAN file
--Collect other physical and experimental parameters from ohcalcs
--Integrate ODE describing population in quantum states, modified from
loadHITRAN module.
--Assume no properties change over feature width
'''

'''
TODO: is basically saturating each IR bin accurate? -- notice that there's no stimulated emission once moves on -- accurate??
TODO: make solveode more efficient:
--only look at relevant part of abfeat.abs_freq based on ir width
--change total integration time on-the-fly based on IR period
TODO: pass KineticsRun parameters to Abs.makeProfile
--

We don't expect velocities to be redistributed when bits of population are
hanging out in the excited state, because correlation time for dilute gas has
limit of infinity -- consequence of low density (Thanks, stat mech II!). But,
there are still some collisions, hence collisional broadening...?
TODO: calculate collisional frequency compared to 10 kHz

New to popmodel3:
-consider populations both within and without rotational level of interest.
-turn off UV laser calculations an option to save memory
"""

import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from scipy.constants import k as kb
from scipy.constants import c, N_A, pi
import loadHITRAN
import ohcalcs as oh
import atmcalcs as atm
from scipy.integrate import ode
from math import floor

# remember to look out for sharing mutable data: https://docs.python.org/2/tutorial/classes.html#class-and-instance-variables
class Sweep(object):
    '''defines a sweeping of the IR laser'''
    def __init__(self,stype='saw',tsweep=1.e-4,irwidth=500.e6,binwidth=1.e6, factor=.1,keepTsweep=False,keepIrwidth=False):
        # Invariant parameters
        self.tsweep=tsweep # s
        self.ircen=0 # set center of swept ir
        self.stype=stype # allowed: 'saw' or 'sin'
        self.binwidth=binwidth

        # Sweep width parameters -- default is max
        self.irwidth=irwidth # Hz, max is 500 MHz, rated by Lockheed Aculight
        self.makebins() # makes initial las_bins array
        self.factor=factor

        # Determines whether alignbins readjusts sweep parameters or not
        self.keepTsweep=keepTsweep
        self.keepIrwidth=keepIrwidth

    def makebins(self):
        self.las_bins = np.arange(self.ircen-self.irwidth/2,self.ircen+self.irwidth/2+self.binwidth,self.binwidth)

    def alignBins(self, abfeat):
        if self.keepIrwidth==False:
            # look at absorption feature to determine size of ir sweeping
            start = np.where(abfeat.pop>self.factor*np.max(abfeat.pop))[0][0]
            end = np.where(abfeat.pop>self.factor*np.max(abfeat.pop))[0][-1] # or could use symmetry...
            abswidth = abfeat.abs_freq[end]-abfeat.abs_freq[start]
            if abswidth > self.irwidth: # keep at maximum if abs feature broad
                print 'alignBins: IR sweep width maximized: 500 MHz'
                abmid = floor(np.size(abfeat.abs_freq)/2.)
                self.las_bins=abfeat.abs_freq[abmid-250:abmid+250]
                abfeat.intpop=abfeat.pop[abmid-250:abmid+250]
            else:
                fullirwidth=self.irwidth
                if self.keepTsweep==False:
                    self.tsweep=self.tsweep*abswidth/fullirwidth # scale sweep time by reduction in new scan width
                    print 'alignBins: IR sweep time reduced to {:.2g} s'.format(self.tsweep)
                else:
                    print 'alignBins: IR sweep time maintained at {:.2g} s'.format(self.tsweep)
                self.irwidth=abswidth
                self.las_bins = abfeat.abs_freq[start:end]
                abfeat.intpop=abfeat.pop[start:end] # portion of population considered in solveode
                print 'alignBins: IR sweep width reduced to {:.2g} MHz'.format(abswidth/1e6)

        else:
            # Keep initial irwidth, but still align bins to abfeat.abs_freq bins
            print 'alignBins: maintaining manual irwidth and tsweep'
            start = np.where(abfeat.abs_freq>=self.las_bins[0])[0][0]
            end = np.where(abfeat.abs_freq<=self.las_bins[-1])[0][-1]
            self.las_bins=abfeat.abs_freq[start:end]
            self.irwidth=self.las_bins[-1]-self.las_bins[0]+self.binwidth
            abfeat.intpop=abfeat.pop[start:end] # portion of population considered in solveode
            print 'alignBins: IR sweep width {:.2g} MHz, sweep time {:.2g} s'.format(self.irwidth/1e6, self.tsweep)
        
        # report how much of the b<--a feature is being swept over:
        part_swept=np.sum(abfeat.intpop)
        print 'alignBins: region swept by IR beam represents {:.1%} of feature\'s total population'.format(part_swept)

class Abs(object):
    '''absorbance line profile, consisting of two 1D arrays:
    abs_freq : bins of frequencies
    pop : relative population absorbing in each frequency bin
    '''
    def __init__(self,wnum,binwidth=1.e6):
        self.wnum=wnum # cm^-1
        self.freq=wnum*c*100 # Hz
        self.binwidth=binwidth
       
    def __str__(self):
        return 'Absorbance feature centered at '+str(self.wnum)+' cm^-1'
      
    def makeProfile(self,abswidth=1000.e6,press=oh.op_press,T=oh.temp,g_air=oh.g_air,mass=oh.mass):
        '''
        Use voigt func to create IR profile we want.
    
        Parameters:
        -----------
        press : float
        Operating pressure, torr. Defaults to ohcalcs value
    
        temp : float
        Temperature. Defaults to ohcalcs value
        '''
        sigma=(kb*T / (mass*c**2))**(0.5)*self.freq # Gaussian std dev
    
        gamma=(g_air*c*100) * press/760. # Lorentzian parameter
        # air-broadened HWHM at 296K, HITRAN (converted from cm^-1 atm^-1)
        # More correctly, correct for temperature -- see Dorn et al. Eq 17

        # Make profile
        enoughWidth=False # Check that we're getting the full profile
        while enoughWidth==False:
            abs_freq = np.arange(-abswidth/2, abswidth/2+self.binwidth, self.binwidth)
            raw_pop=voigt(abs_freq,1,0,sigma,gamma,True)
            norm_factor = 1/np.sum(raw_pop)
            pop=raw_pop * norm_factor # makes sum of pops = 1.
            # Expand feature if population isn't negligible at edge of abs_freq.
            if pop[0]>=0.01*np.max(pop):
                abswidth=abswidth*1.5
            else:
                enoughWidth=True
        self.abs_freq = abs_freq
        self.pop = pop
        startfwhm=np.where(pop>=np.max(pop)*0.5)[0][0]
        endfwhm=np.where(pop>=np.max(pop)*0.5)[0][-1]
        fwhm=abs_freq[endfwhm]-abs_freq[startfwhm]
        print'makeProfile: made abs profile'
        print'makeProfile: abs profile has FHWM = {:.2g} MHz'.format(fwhm/1e6)
        print'makeProfile: total width of stored array = {:.2g} MHz'.format(abswidth/1e6)
        # return np.array([abs_freq, pop])

def voigt(xarr,amp,xcen,sigma,gamma,normalized=False):
    """
    From pyspeckit, on Github.
    Normalized Voigt profile

    z = (x+i*gam)/(sig*sqrt(2))
    V(x,sig,gam) = Re(w(z))/(sig*sqrt(2*pi))

    The area of V in this definition is 1.
    If normalized=False, then you can divide the integral of V by
    sigma*sqrt(2*pi) to get the area.

    Original implementation converted from
    http://mail.scipy.org/pipermail/scipy-user/2011-January/028327.html
    (had an incorrect normalization and strange treatment of the input
    parameters)

    Modified implementation taken from wikipedia, using the definition.
    http://en.wikipedia.org/wiki/Voigt_profile

    Parameters
    ----------
    xarr : np.ndarray
    The X values over which to compute the Voigt profile
    amp : float
    Amplitude of the voigt profile
    if normalized = True, amp is the AREA
    xcen : float
    The X-offset of the profile
    sigma : float
    The width / sigma parameter of the Gaussian distribution -- standard deviation
    gamma : float
    The width / shape parameter of the Lorentzian distribution -- HWHM
    normalized : bool
    Determines whether "amp" refers to the area or the peak
    of the voigt profile
    """
    z = ((xarr-xcen) + 1j*gamma) / (sigma * np.sqrt(2))
    V = amp * np.real(scipy.special.wofz(z))
    if normalized:
        return V / (sigma*np.sqrt(2*np.pi))
    else:
        return V

class KineticsRun(object):
    '''Entire model of OH population kinetics. Has single instance of Sweep
    and of Abs. Sweep is made in __init__, while Abs is made after the HITRAN
    file is imported and the absorption feature selected.'''
    def __init__(self,
        press=oh.op_press,
        temp=oh.temp,
        xoh=oh.xoh,
        stype='sin',
        delay=100,
        keepN=False,
        tsweep=1.e-4,
        irwidth=500.e6,
        binwidth=1.e6,
        factor=.1,
        keepTsweep=False,
        keepIrwidth=False,
        withoutUV=False,
        rotequil=True,
        redistequil=True):
        # Sweep object
        self.sweep=Sweep(stype=stype,
                        tsweep=tsweep,
                        irwidth=irwidth,
                        binwidth=binwidth,
                        factor=factor,
                        keepTsweep=keepTsweep,
                        keepIrwidth=keepIrwidth)

        # Operation parameters
        self.press=press # torr
        self.temp=temp # K
        self.xoh=xoh # mixing ratio, to account for amt of signal increasing with op_press
        self.ohtot = atm.press_to_numdens(press,temp)*xoh

        # ODE solution parameters
        self.delay = delay # s, artificial UV delay in model for a and b to equilibrate.
        # delay currently huge to effectively 'turn off' UV pulse for now...
        self.keepN = keepN # keep individual N values or not -- 'True' causes memory error if N is too large.
        self.withoutUV = withoutUV # don't bother with UV calcs if not needed
        self.rotequil = rotequil
        self.redistequil = redistequil
        
    def addhitran(self,file,a):
        # Collect info from HITRAN and extract:
        self.hpar = loadHITRAN.processHITRAN(file)
        self.hline = self.hpar[a] # single set of parameters from hpar

    def solveode(self, file='13_hit12.par', a=24, intperiods=2.1, avg_step_in_bin=20.):
        '''Integrate ode describing IR-UV two-photon LIF, given master equation
        (with its Jacobian) and all relevant parameters. Use 'ohcalcs' and
        'atmcalcs' modules.
        
        Define global parameters that are independent of HITRAN OH IR data within
        function: Additional OH parameters related to 'c' state and quenching, and
        laser parameters. Also set up parameters for solving and plotting ODE.

        Parameters:
        -----------
        file : str
        Input HITRAN file (160-char format).
        
        a : int
        index of transition to use, within 'file'. a=24 gives Q1(1) line in '13_hit12.par'

        Outputs:
        --------
        N : ndarray
        Relative population of 'a', 'b' and 'c' states over integration time.    
        '''
        print 'solveode: integrating at {} torr, {} K, OH in cell {:.2g} cm^-3'.format(self.press,self.temp,self.ohtot)
        print 'solveode: sweep mode: {}'.format(self.sweep.stype)
        # Collect info from HITRAN and extract:
        self.addhitran(file,a)
        print 'solveode: using {}({}) line at {:.4g} cm^-1'.format(self.hline['branch'],self.hline['line'],self.hline['wnum_ab'])

        # Set up IR b<--a absorption profile
        self.abfeat = Abs(wnum=self.hline['wnum_ab'])
        self.abfeat.makeProfile(press=self.press,
                                T=self.temp,
                                g_air=self.hline['g_air'])
        
        # Algin bins for IR laser and absorbance features for integration
        self.sweep.alignBins(self.abfeat)

        # set integration time based on IR sweep time
        tl = self.sweep.tsweep*intperiods # total integration time
        avg_bintime = self.sweep.tsweep/(2*self.sweep.irwidth/self.sweep.binwidth) # for 'sin'. Twice as long for 'saw'
        dt = avg_bintime/avg_step_in_bin # avg 20 steps within each ir bin -- checked for agreement with 100 steps/bin
        self.tbins = np.arange(0, tl+dt, dt)
        t_steps = np.size(self.tbins)

        print 'solveode: integrating {:.2g} s, step size {:.2g} s'.format(tl,dt)

        # define some local variables to make more succinct
        num_las_bins=np.size(self.sweep.las_bins)
        num_int_bins=num_las_bins+2 # add bins for not covered by laser, other rot levels
        tsweep = self.sweep.tsweep
        stype = self.sweep.stype


        # Determine location of swept IR (a to b) laser by defining 'sweepfunc' 1D array attribute:
        # Stores index of las_bins corresponding to each index in tsweep
        tindex=np.arange(np.size(self.tbins))
        tindexsweep=np.searchsorted(self.tbins,tsweep,side='right')-1
        if stype=='saw':
            self.sweepfunc=np.floor((tindex%tindexsweep)*(num_las_bins)/tindexsweep)
        elif stype=='sin':
            self.sweepfunc = np.round((num_las_bins-1)/2.*np.sin(2*pi/tindexsweep*tindex)+(num_las_bins-1)/2.)

        # set up ODE
        # Initial state N0 with pop distribution in ground state
        if self.withoutUV == False:
            self.nlevels=3
        else:
            self.nlevels=2    
        # assume for now dealing with N"=1.
        self.N0 = np.zeros((self.nlevels,num_int_bins))
        self.N0[0,0:-2] = self.abfeat.intpop * oh.rotfrac[0] * self.ohtot
        self.N0[0,-2] = (self.abfeat.pop.sum() - self.abfeat.intpop.sum()) * oh.rotfrac[0] * self.ohtot # feature pop not covered by laser
        self.N0[0,-1] = self.ohtot * (1-oh.rotfrac[0]) # population in other rotational levels

        # Create array to store integrated values, depending on keepN
        if self.keepN == True:
            # N is array to populate with a/b/c state populations integrated over time.
            self.N=np.empty((t_steps,self.nlevels,num_int_bins))
            self.N[0] = self.N0
        elif self.keepN == False:
            # just keep population in a/b/c, summed over width of feature. Distinguish within/out rot level of interest.
            self.abcpop=np.empty((t_steps,self.nlevels,2))
            self.abcpop[0] = np.array([self.N0[:,0:-1].sum(1),self.N0[:,-1]]).T

        # Integration object using scipy.integrate
        r = ode(self.dN)
        r.set_integrator('lsoda', with_jacobian=False)
        r.set_initial_value(list(self.N0.ravel()), 0)

        print '  %  |   time   |   bin   '
        print '--------------------------'

        # Solve ODE
        old_complete=0
        self.time_progress=0 # used within dN to determine index of sweepfunc to use.
        while r.successful() and r.t < tl-dt:
            # display progress
            complete = r.t/tl

            if floor(complete*100/10)!=floor(old_complete*100/10):
                print ' {0:>3.0%} | {1:8.2g} | {2:7.0f} '.format(complete, r.t, self.sweepfunc[self.time_progress])
            old_complete = complete
            
            # integrate
            entry=int(round(r.t/dt))+1
            nextstep = r.integrate(r.t + dt)
            nextstepN = np.resize(nextstep, (self.nlevels,num_int_bins))

            # save output
            if self.keepN == True:
                self.N[entry] = nextstepN
            else:
                self.abcpop[entry] = np.array([nextstepN[:,0:-1].sum(1),nextstepN[:,-1]]).T

            self.time_progress+=1

        print 'solveode: done with integration'

    def dN(self, t, y):
        '''differential equations describing three-state model for OH pops'''

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

        # Determine position of IR laser
        voigt_pos = self.sweepfunc[self.time_progress]
        num_las_bins=np.size(self.sweep.las_bins)
        num_int_bins=num_las_bins+2
        if voigt_pos+1 > num_las_bins:
            print 'dN: WARNING: voigt_pos out of range'
        Lab_sweep=np.zeros(num_int_bins)
        Lab_sweep[voigt_pos]=Lab

        # reshape y back into form where each nested 1D array contains all
        # populations in given energy level:
        y=y.reshape(self.nlevels,-1)
        
        # Coefficients related to...
        # b<--a processes:
        absorb_ab = Bab * Lab_sweep * y[0]

        # b-->a processes: (spont emission negligible)
        stim_emit_ba = Bba * Lab_sweep * y[1]
        quench_b = kqb * Q * y[1]

        # rotational relaxation:
        # convention that positive value = increase pop in single rot state of interest
        rrout = np.array([7.72e-10,7.72e-10, 4.65e-10]) # Smith and Crosley, 1990. Undifferentiated for quencher or vibrational state
        rrin = rrout * oh.rotfrac/(1-oh.rotfrac)

        if self.redistequil:
            fdist = (self.N0[0,0:-1]/self.N0[0,0:-1].sum())
        elif y[0,0:-1].sum() != 0:
            fdist = (y[0,0:-1]/y[0,0:-1].sum())
        else:
            fdist = 0

        # if UV laser is on:
        if self.withoutUV == False:
            # Pulse the UV (b to c) laser (assume total modeled time < rep rate):
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

            dN0 = - absorb_ab + stim_emit_ba + spont_emit_ca + quench_b + quench_c
            dN1 = absorb_ab - stim_emit_ba + stim_emit_cb - absorb_bc - quench_b + spont_emit_cb
            dN2 = - spont_emit_cb + absorb_bc - spont_emit_ca - quench_c - stim_emit_cb

            intermediate = np.array([dN0, dN1, dN2])

            rrvalues = np.empty_like(intermediate)
            if self.rotequil:
                rrvalues[0,0:-1] = -y[0,0:-1]*Q*rrout[0] + y[0,-1]*Q*rrin[0]*fdist
                # assuming repopulation from other rotational levels flows to rot level of interest based on equilibrium distribution (at least for ground)
                rrvalues[0,-1] = y[0,0:-1].sum()*Q*rrout[0] - y[0,-1]*Q*rrin[0]
                
                if y[1,0:-1].sum() != 0: # avoid divide by zero error
                    rrvalues[1,0:-1] = -y[1,0:-1]*Q*rrout[1] + y[1,-1]*Q*rrin[1]*fdist
                    # assume same equilibrium distribution as v"=0 -- probably not a terrible approximation
                    rrvalues[1,-1] = y[1,0:-1].sum()*Q*rrout[1] - y[1,-1]*Q*rrin[1]
                else:
                    rrvalues[1,:] = 0

                if y[2,0:-1].sum() != 0: # avoid divide by zero error
                    rrvalues[2,0:-1] = -y[2,0:-1]*Q*rrout[2] + y[2,-1]*Q*rrin[2]*fdist
                    rrvalues[2,-1] = y[2,0:-1].sum()*Q*rrout[2] - y[2,-1]*Q*rrin[2]
                else:
                    rrvalues[2,:] = 0

            else:
                rrvalues.fill(0)

            result = (intermediate + rrvalues).ravel()
            # flatten to 1D array: 1st all entries for 'a' states, 2nd all for 'b', ...:
            return result
        
        else:
            dN0 = - absorb_ab + stim_emit_ba + quench_b
            dN1 = absorb_ab - stim_emit_ba - quench_b
            intermediate = np.array([dN0, dN1])

            rrvalues = np.empty_like(intermediate)
            if self.rotequil:
                rrvalues[0,0:-1] = -y[0,0:-1]*Q*rrout[0] + y[0,-1]*Q*rrin[0]*fdist
                # assuming repopulation from other rotational levels flows to rot level of interest based on equilibrium distribution (at least for ground)
                rrvalues[0,-1] = y[0,0:-1].sum()*Q*rrout[0] - y[0,-1]*Q*rrin[0]
                
                if y[1,0:-1].sum() != 0: # avoid divide by zero error
                    rrvalues[1,0:-1] = -y[1,0:-1]*Q*rrout[1] + y[1,-1]*Q*rrin[1]*fdist
                    # assume same equilibrium distribution as v"=0 -- probably not a terrible approximation
                    rrvalues[1,-1] = y[1,0:-1].sum()*Q*rrout[1] - y[1,-1]*Q*rrin[1]
                else:
                    rrvalues[1,:] = 0
            else:
                rrvalues.fill(0)
            result = (intermediate + rrvalues).ravel()

            return result
        
    def plotpops(self):
        '''Given solution N to solveode, plot 'b' state population over time.

        Requires:
        -either 'abcpop' or 'N' from solveode input (makes 'abcpop' from 'N' if missing)
            -if 'abcpop' is made, requires tbins, nlevels,

        '''
        if hasattr(self,'abcpop')==False and hasattr(self,'N')==False:
            print 'need to run solveode first!'
            return
        elif hasattr(self,'abcpop')==False and hasattr(self,'N')==True:
            self.abcpop = np.empty((np.size(self.tbins),self.nlevels,2))
            self.abcpop[:,:,0]=self.N[:,:,0:-1].sum(2)
            self.abcpop[:,:,1]=self.N[:,:,-1]

        self.plotvlaser(self.abcpop[:,1,0]/self.ohtot,'Relative population in v\"=1, N\"=1, J\"=1.5','Fraction of total OH')

    def plotvlaser(self,func,title='plot',yl='y axis'):
        '''make arbitrary plot in time w laser sweep as second plot'''
        if hasattr(self,'abcpop')==False and hasattr(self,'N')==False:
            print 'need to run solveode first!'
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
        ax1.plot(self.tbins*1e6,self.sweep.las_bins[self.sweepfunc[time_indices].astype(int)]/1e6)
        ax1.set_title('Position of IR beam')
        ax1.set_xlabel('Time ($\mu$s)')
        ax1.set_ylabel('Relative Frequency (MHz)')
        plt.show()    

    def plotfeature(self):
        ''''''
        fig, (ax0) = plt.subplots(nrows=1)
        ax0.plot(self.abfeat.abs_freq,self.abfeat.pop)
        ax0.axvline(self.sweep.las_bins[0],ls='--')
        ax0.axvline(self.sweep.las_bins[-1],ls='--')

    def saveOutput(self,file):
        ''''''
        np.savez(file,
            abcpop=self.abcpop,
            las_bins=self.sweep.las_bins,
            tbins=self.tbins,
            sweepfunc=self.sweepfunc,
            abs_freq=self.abfeat.abs_freq,
            pop=self.abfeat.pop)

    def loadOutput(self,file):
        with np.load(file) as data:
            self.abcpop=data['abcpop']
            self.sweep.las_bins=data['las_bins']
            self.tbins=data['tbins']
            self.sweepfunc=data['sweepfunc']
            self.abfeat = Abs(0)
            self.abfeat.abs_freq=data['abs_freq']
            self.abfeat.pop=data['pop']

def pressdepen(file):
    i=1
    pressconsidered=(2,10,100,760)
    for press in pressconsidered:
        print '--------------------'
        print 'KineticsRun {:} OF {}'.format(i,np.size(pressconsidered))
        print '--------------------'
        k=KineticsRun(press=press,stype='sin')
        k.solveode(file)
        # k.plotpops()
        #k.abfeat=Abs()
        #k.abfeat.makeProfile(press=press)
        #k.sweep.matchAbsSize(k.abfeat)
        i+=1

def sweepdepen(file):
    for factor in (0.01, 0.1, 0.5, 0.9):
        k=KineticsRun(stype='sin')
        k.sweep.factor=factor
        # k.abfeat=Abs()
        # k.abfeat.makeProfile()
        # k.sweep.matchAbsSize(k.abfeat)
        k.solveode(file)
        # k.plotpops()
