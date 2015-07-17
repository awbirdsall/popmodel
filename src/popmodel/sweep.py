'''popmodel module for Sweep class
'''
import numpy as np
import logging

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

        self.logger = logging.getLogger('popmodel.sweep.Sweep')

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
                self.logger.info('call from within Sweep within sweep.py')
                self.logger.info('alignBins: IR sweep width maximized: {:.2g} MHz'
                    .format(self.width/1e6))
                abmid = floor(np.size(abfeat.abs_freq)/2.)
                irfw=self.width/self.binwidth
                self.las_bins=abfeat.abs_freq[abmid-irfw/2:abmid+irfw/2]
                abfeat.intpop=abfeat.pop[abmid-irfw/2:abmid+irfw/2]
            else: # reduce self.width to abswidth
                fullwidth=self.width
                if self.keepTsweep==False: # scale tsweep by width reduction
                    self.tsweep=self.tsweep*abswidth/fullwidth 
                    self.logger.info('alignBins: IR sweep time reduced to '+
                        '{:.2g} s'.format(self.tsweep))
                else:
                    self.logger.info('alignBins: IR sweep time maintained at ,'
                        '{:.2g} s'.format(self.tsweep))
                self.width=abswidth
                self.las_bins = abfeat.abs_freq[start:end]
                abfeat.intpop=abfeat.pop[start:end] # integrated pop
                self.logger.info('alignBins: IR sweep width reduced to {:.2g} MHz'
                    .format(abswidth/1e6))

        else:
            # Keep initial width, but still align bins to abfeat.abs_freq
            self.logger.info('alignBins: maintaining manual width and tsweep')
            start = np.where(abfeat.abs_freq>=self.las_bins[0])[0][0]
            end = np.where(abfeat.abs_freq<=self.las_bins[-1])[0][-1]
            self.las_bins=abfeat.abs_freq[start:end]
            self.width=self.las_bins[-1]-self.las_bins[0]+self.binwidth
            abfeat.intpop=abfeat.pop[start:end] # integrated pop
            self.logger.info('alignBins: sweep width ',
                '{:.2g} MHz, sweep time {:.2g} s'.format(self.width/1e6,
                    self.tsweep))
        # report how much of the b<--a feature is being swept over:
        self.part_swept=np.sum(abfeat.intpop)
        self.logger.info('alignBins: region swept by IR beam represents '+
            '{:.1%} of feature\'s total population'.format(self.part_swept))
