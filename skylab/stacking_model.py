import ps_model
import numpy as np

_2dim_bins = 25

PowerLawLLH = ps_model.PowerLawLLH


class StackingLLH(PowerLawLLH):
    r"""Likelihood using Energy Proxy and declination, where declination is
    used for normalisation to account for changing energy distributions.

    """
    def __init__(self, twodim_bins=_2dim_bins, twodim_range=None,
                 **kwargs):
        r"""Constructor

        """
        super(StackingLLH, self).__init__(["logE", "sinDec"],
                                        twodim_bins, range=twodim_range,
                                        **kwargs)




        return
    
    def signal(self, src_ra, src_dec, ev,  src_sigma):
        r"""Spatial distance between source position and events

        Signal is assumed to cluster around source position.
        The distribution is assumed to be well approximated by a gaussian
        locally.

        Parameters
        -----------
        ev : structured array
            Event array, import information: sinDec, ra, sigma

        Returns
        --------
        P : array-like
            Spatial signal probability for each event

        """
        cos_ev = np.sqrt(1. - ev["sinDec"]**2)
        
        #radiffs = src_ra[:, np.newaxis]-ev["ra"]
        #second = np.cos(src_dec[:, np.newaxis]) * cos_ev

        
        #distold = np.arccos(np.cos(src_ra[0] - ev["ra"])
                            #* np.cos(src_dec[0]) * cos_ev
                         #+ np.sin(src_dec[0]) * ev["sinDec"])
	
	dist = np.arccos(np.cos(src_ra[:, np.newaxis]-ev["ra"])
                            * np.cos(src_dec[:, np.newaxis]) * cos_ev
                         + np.sin(src_dec[:, np.newaxis])*ev["sinDec"])
	

	
	

        return (1./2./np.pi/(ev["sigma"]**2 + src_sigma[:,np.newaxis]**2)
                * np.exp(-dist**2 / 2. / (ev["sigma"]**2 + src_sigma[:,np.newaxis]**2)))
     