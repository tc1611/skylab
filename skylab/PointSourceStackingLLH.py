import psLLH
from ps_injector import PointSourceInjector
import numpy as np
from copy import deepcopy
import logging
import healpy as hp
import numpy as np
import numpy.lib.recfunctions
import scipy.interpolate
import scipy.optimize
import scipy.stats
from scipy.signal import convolve2d

_precision = (4.-1.)/30.
_gamma_bins = np.linspace(1., 4., 30 + 1)
_pgtol = 1.e-3
_gamma_params = dict(gamma=[2., (1., 4.)])
_aval = 1.e-3
_rho_max = 0.95

def trace(self, message, *args, **kwargs):
    r"""Add trace to logger with output level beyond debug

    """
    if self.isEnabledFor(5):
        self._log(5, message, args, **kwargs)

logging.addLevelName(5, "TRACE")
logging.Logger.trace = trace

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

PointSourceLLH = psLLH.PointSourceLLH

class PointSourceStackingLLH(PointSourceLLH):
    _SrcWeightsDict = {}
    _SrcWeightsTable ={}
    
    _precision = _precision
    
    def _around(self, value):
        r"""Round a value to a precision defined in the class.

        Parameters
        -----------
        value : array-like
            Values to round to precision.

        Returns
        --------
        round : array-like
            Rounded values.

        """
        return _gamma_bins[((value - _gamma_bins).clip(min=0)).argmin()]
    
    def makeSrcWeightsTable(self, src_dec, src_weights, arr_mc):
      
	src_weights = src_weights/src_weights.sum()
	SrcWeightsDict={}
	injectordict={}
	print "Now constructing gamma dependent source weights table for stacking :"
	for gamma in _gamma_bins:
	  injectordict[gamma] = {}
	  SrcWeightsDict[gamma]=np.zeros(len(src_dec))
	  for dec in src_dec:
	      injectordict[gamma][dec] = PointSourceInjector(gamma=gamma)
	      injectordict[gamma][dec].fill(np.deg2rad(dec), arr_mc)
	

	
	SrcWeightsTable = np.empty([len(src_dec), len(_gamma_bins)])
	
	for i in range(0, len(src_dec)):
	  for j in range(0, len(_gamma_bins)):
	    SrcWeightsTable[i][j] = injectordict[_gamma_bins[j]][src_dec[i]].flux2mu(1.)*src_weights[i]
	    SrcWeightsDict[_gamma_bins[j]][i] = SrcWeightsTable[i][j]
	
	for gamma in _gamma_bins:
	  sumrow = 0.
	  for i in range(0, len(src_dec)):
	    sumrow = sumrow + SrcWeightsDict[gamma][i]
	  SrcWeightsDict[gamma] = SrcWeightsDict[gamma]/sumrow    

	    
	#print SrcWeightsTable
	
	self._SrcWeightsDict = SrcWeightsDict
	
	for key in sorted(self._SrcWeightsDict.keys()):
	  print key, self._SrcWeightsDict[key]
	
	return SrcWeightsTable
  
  
    def _select_events(self, src_ra, src_dec, src_sigma, **kwargs):
        r"""Select events around source location(s) used in llh calculation.

        Parameters
        ----------
        src_ra src_dec : float, array_like
            Rightascension and Declination of source(s)

        Other parameters
        ----------------
        scramble : bool
            Scramble rightascension prior to selection.
        inject : numpy_structured_array
            Events to add to the selected events, fields equal to exp. data.

        """

        scramble = kwargs.pop("scramble", False)
        inject = kwargs.pop("inject", None)
        stack = kwargs.pop("stack", False)
        
	if stack:
	    logger.info("When Stacking, all events are selected automatically")
	    self.mode = "all"
        
        if kwargs:
            raise ValueError("Don't know arguments", kwargs.keys())

        # get the zenith band with correct boundaries
        #dec = (np.pi - 2. * self.delta_ang) / np.pi * src_dec
        #min_dec = max(-np.pi / 2., dec - self.delta_ang)
        #max_dec = min(np.pi / 2., dec + self.delta_ang)

        dPhi = 2. * np.pi


        exp_mask = np.ones_like(self.exp["sinDec"], dtype=np.bool)


        # update the zenith selection and background probability
        ev = self.exp[exp_mask]

        # update rightascension information for scrambled events
        if scramble:
            ev["ra"] = self.random.uniform(0., 2. * np.pi, size=len(ev))

        # selection in rightascension

        if inject is not None:
            '''
            # how many events are randomly inside of the selection
            m = (self.random.poisson(float(len(inject)) * dPhi / (2.*np.pi))
                    if self.mode == "box" else len(inject))
            ind = self.random.choice(len(ev), size=min(m, len(ev)),
                                     replace=False)

            ev = ev[np.arange(len(ev))[np.in1d(np.arange(len(ev)), ind,
                                               invert=True)]]
            '''

            ev = np.append(ev, numpy.lib.recfunctions.append_fields(
                                        inject, ["B", "S"],
                                        [self.llh_model.background(inject),
                                         np.zeros(len(inject))],
                                        usemask=False))

        # calculate signal term
        evsigarray = self.llh_model.signal(src_ra, src_dec, ev, src_sigma)

        # do not calculate values with signal below threshold
        #ev_mask = ev["S"] > self.thresh_S
        #ev = ev[ev_mask]

        # set number of selected events
        n = len(ev)

        if (n < 1
            and (np.sin(src_dec) < self.sinDec_range[0]
                 and np.sin(src_dec) > self.sinDec_range[-1])):
            logger.error("No event was selected, fit will go to -infinity")

        logger.info("Select new events for mode {0:s}\n".format(self.mode) +
                    ("For point at ra = {0:6.2f} deg, dec = {1:-6.2f} deg, " +
                     "{2:6d} events were selected").format(
                          np.degrees(src_ra[0]), np.degrees(src_dec[0]), n))
	
	#print ev
	print evsigarray
	
        return ev, evsigarray
    
    
    def llh(self, src_ra, src_dec, ev, evsigarray, src_weights, **fit_pars):
      
	nsources = fit_pars.pop("nsources")
	gamma = fit_pars['gamma']
	#print "This is the fucking fitpars", fit_pars
	
	n = len(ev)
	N = self.N
	
	w, grad_w = self.llh_model.weight(ev, **fit_pars)
	
        g1 = self._around(gamma)
        dg = self._precision

        # evaluate neighbouring gridpoints and parametrize a parabola
        g0 = self._around(g1 - dg)
        g2 = self._around(g1 + dg)	
        
        W0 = self._SrcWeightsDict[g0]
        W1 = self._SrcWeightsDict[g1]
        W2 = self._SrcWeightsDict[g2]
        
        a = (W0 - 2.*W1 + W2)/(2. *dg**2)
        b = (W2-W0)/(2.*dg)
        
        wval = (a * (gamma - g1)**2 + b * (gamma - g1) + W1)
        
        #print gamma, g0, g1, g2
        
        wval = wval/wval.sum()
        
        #print wval
        
        wgrad = wval * (2. * a * (gamma - g1) + b)
        
        SoB = np.dot(wval, evsigarray) / ev["B"]
        
        #print "Sum", SoB.sum()
        
        wSoB = np.dot(wgrad, evsigarray) / ev["B"]
        
        w, grad_w = self.llh_model.weight(ev, **fit_pars)
	
        x = (SoB * w - 1.) / N

        # check which sums of the likelihood are close to the divergence

        
        aval = -1. + _aval
        alpha = nsources * x

        # select events close to divergence
        xmask = alpha > aval

        # function value, log1p for OK, otherwise quadratic taylor
        funval = np.empty_like(alpha, dtype=np.float)
        funval[xmask] = np.log1p(alpha[xmask])
        funval[~xmask] = (np.log1p(aval)
                      + 1. / (1.+aval) * (alpha[~xmask] - aval)
                      - 1./2./(1.+aval)**2 * (alpha[~xmask]-aval)**2)
        funval = funval.sum() + (N - n) * np.log1p(-nsources / N)

        # gradients

        # in likelihood function
        ns_grad = np.empty_like(alpha, dtype=np.float)
        ns_grad[xmask] = x[xmask] / (1. + alpha[xmask])
        ns_grad[~xmask] = (x[~xmask] / (1. + aval)
                       - x[~xmask] * (alpha[~xmask] - aval) / (1. + aval)**2)
        ns_grad = ns_grad.sum() - (N - n) / (N - nsources)

        # in weights
        if grad_w is not None:
            par_grad = 0.5 / N * SoB * grad_w + 0.5 / N * wSoB

            par_grad[:, xmask] *= nsources / (1. + alpha[xmask])
            par_grad[:, ~xmask] *= (nsources / (1. + aval)
                                    - nsources * (alpha[~xmask] - aval)
                                        / (1. + aval)**2)

            par_grad = par_grad.sum(axis=-1)

        else:
            par_grad = np.zeros((0,))

        grad = np.append(ns_grad, par_grad)

        # multiply by two for chi2 distributed test-statistic
        LogLambda = 2. * funval
        grad = 2. * grad

        return LogLambda, grad
	
      
    def fit_stack(self, src_ra, src_dec, src_sigma, src_weight, **kwargs):
	def _llh(x, *args):
            """Scale likelihood variables so that they are both normalized.
            Returns -logLambda which is the test statistic and should
            be distributed with a chi2 distribution assuming the null
            hypothesis is true.

            """

            fit_pars = dict([(par, xi) for par, xi in zip(self.params, x)])

            fun, grad = self.llh(src_ra, src_dec, ev, evsigarray, src_weight, **fit_pars)

            # return negative value needed for minimization
            return -fun, -grad
	
	self.makeSrcWeightsTable(src_dec, src_weight, self.mc)
	
        scramble = kwargs.pop("scramble", False)
        inject = kwargs.pop("inject", None)
        stack = kwargs.pop("stack", True)
        kwargs.setdefault("pgtol", _pgtol)
        
        ev, evsigarray = self._select_events(src_ra, src_dec, src_sigma, inject=inject, scramble=scramble, stack=stack) 
        
        n = (len(ev) if not isinstance(ev, dict)
                     else sum([len(ev_i) for ev_i in ev.itervalues()]))

        # get seeds
        pars = self.par_seeds
        inds = [i for i, par in enumerate(self.params) if par in kwargs]
        pars[inds] = np.array([kwargs.pop(par) for par in self.params
                                               if par in kwargs])

        # minimizer setup
        xmin, fmin, min_dict = scipy.optimize.fmin_l_bfgs_b(
                                _llh, pars,
                                bounds=self.par_bounds,
                                **kwargs)

        if abs(xmin[0]) > _rho_max * n:
            logger.error(("nsources > {0:7.2%} * {1:6d} selected events, " +
                          "fit-value nsources = {2:8.1f}").format(
                              _rho_max, n, xmin[0]))

        xmin = dict([(par, xi) for par, xi in zip(self.params, xmin)])

        # Separate over and underfluctuations
        fmin *= -np.sign(xmin["nsources"])

        return fmin, xmin