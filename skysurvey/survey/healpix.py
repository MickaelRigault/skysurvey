from .core import Survey

import pandas
import numpy as np
import healpy as hp



def get_ipix_in_range(nside, ra_range=None, dec_range=None, in_rad=False):
    """ """
    npix = hp.nside2npix(nside)
    pixs = np.arange(npix) # list of all healpix pixels
    if ra_range is None and dec_range is None:
        return pixs
    
    ras,decs = hp.pix2ang(nside, pixs)
    ras = (np.pi/2-ras)
    # only dec range
    if ra_range is None:
        if not in_rad:
            dec_range = np.multiply(dec_range, np.pi/180) # works if list given
        return pixs[(decs>=dec_range[0]) & (decs<=dec_range[1])]
    
    # only ra range    
    if dec_range is None:
        if not in_rad:
            ra_range = np.multiply(ra_range, np.pi/180) # works if list given
        return pixs[(ras>=ra_range[0]) & (ras<=ra_range[1])]
    
    # both
    if not in_rad:
        ra_range = np.multiply(ra_range, np.pi/180) # works if list given
        dec_range = np.multiply(dec_range, np.pi/180) # works if list given
    return pixs[(ras>=ra_range[0]) & (ras<=ra_range[1]) & (decs>=dec_range[0]) & (decs<=dec_range[1])]


# ================== #
#                    #
#    Healpix         #
#                    #
# ================== #
class HealpixSurvey( Survey ):
    
    def __init__(self, nside, data=None):
        """ """
        super().__init__(data)
        self._nside = nside
        
    @classmethod
    def from_data(cls, nside, data):
        """ """
        return cls(nside=nside, data=data)

    @classmethod
    def from_random(cls, nside, size, 
                    bands,  
                    mjd_range, skynoise_range, 
                    ra_range=None, dec_range=None):
        """ """
        this = cls(nside=nside)
        this.draw_random(size,  bands,  
                        mjd_range, skynoise_range, 
                        ra_range=ra_range, dec_range=dec_range,
                        inplace=True)
        return this
    
    # ============== #
    #   Methods      #
    # ============== #
    def draw_random(self, size, 
                    bands, mjd_range, skynoise_range, gain_range=1,
                    ra_range=None, dec_range=None,
                    inplace=False, nside=None):
        """ """
        if nside is None: # don't change nside
            nside = self.nside
        elif inplace: # change nside
            warnings.warn("Cannot change nside with inplace=True, a copy (inplace=False) is returned.")
            inplace = False
            
        data = self._draw_random(nside, size, 
                                 bands, mjd_range, skynoise_range, 
                                 ra_range=ra_range, dec_range=dec_range,
                                 gain_range=gain_range)
        
        if not inplace:
            return self.__class__.from_data(nside=nside, data=data)

        self.set_data(data)
        
        
    def radec_to_fieldid(self, ra, dec):
        """ """
        return hp.ang2pix(self.nside, (90 - dec) * np.pi/180, ra * np.pi/180)
        
    # ----------- #
    #  PLOTTER    #
    # ----------- #
    def show(self, stat='size', column=None, title=None, data=None, **kwargs):
        """ shows the sky coverage """
        data = self.get_fieldstat(stat=stat, columns=column, incl_zeros=True, fillna=np.NaN, data=data)
        hp.mollview(data, title=title, **kwargs)
        
    # ============== #
    # Static Methods #
    # ============== #        
    @staticmethod
    def _draw_random(nside, size, 
                     bands,  
                     mjd_range, skynoise_range,
                     gain_range=1,
                     ra_range=None, dec_range=None):
        """ 
        *_range can be 2d-array [min, max] or single values. 
        """
        # np.resize(1, 2) -> [1,1]
        mjd = np.random.uniform(*np.resize(mjd_range,2), size=size)
        band = np.random.choice(bands, size=size)
        skynoise = np.random.uniform(*np.resize(skynoise_range, 2), size=size)
        gain = np.random.uniform(*np.resize(gain_range, 2), size=size)
        # = coords
        # no radec limit
        if ra_range is None and dec_range is None:
            npix = hp.nside2npix(nside)
            ipix = np.random.uniform(0, npix, size=size)
        else:
            ipix_ok = get_ipix_in_range(nside, ra_range=ra_range, dec_range=dec_range)
            ipix = np.random.choice(ipix_ok, size=size)
            
        # data sorted by mjd
        data = pandas.DataFrame(zip(mjd, band, skynoise, gain, ipix),
                               columns=["mjd","band","skynoise", "gain", "fieldid"]
                               ).sort_values("mjd"
                               ).reset_index(drop=False) # don't need to know the creation order
        return data
    
    # ============== #
    #   Properties   #
    # ============== #
    @property
    def nside(self):
        """ dataframe containing what has been observed when """
        return self._nside
    
    @property
    def nfields(self):
        """ shortcut to npix """
        return self.npix
    
    @property    
    def npix(self):
        """ number of healpix pixels """
        if not hasattr(self, "_npix") or self._npix is None:
            self._npix = hp.nside2npix(self.nside)
            
        return self._npix
    
    
    @property
    def metadata(self):
        """ """
        meta = super().metadata
        meta["nside"] = self.nside
        return meta
