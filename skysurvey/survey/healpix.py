from .core import BaseSurvey

import pandas
import numpy as np
import healpy as hp


__all__ = ["HealpixSurvey"]


def get_ipix_in_range(nside, ra_range=None, dec_range=None, in_rad=False):
    """ get the healpix pixel index (ipix) that are with a given ra and dec range

    Parameters
    ----------
    ra_range, dec_range: 2d-array, None
        min and max to define a coordinate range to be considered.
        None means no limit.

    in_rad: bool
        are the ra and dec coordinates in radian (True)
        or degree (False)

    Returns
    -------
    list
        list of healpix pixel index ipix
    """
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
class HealpixSurvey( BaseSurvey ):
    
    def __init__(self, nside, data=None):
        """ 
        See also
        --------
        from_data: loads the instance given observing data.
        from_random: generate random observing data and loads the instance.
        """
        super().__init__(data)
        self._nside = nside
        
    @classmethod
    def from_data(cls, nside, data):
        """ load an instance given survey data and healpix size (nside) 
        
        Parameters
        ----------
        nside : int
            healpix nside parameter

        data: pandas.DataFrame
            observing data.

        Returns
        -------
        class instance

        See also
        --------
        from_random: generate random observing data and loads the instance.
        
        """
        return cls(nside=nside, data=data)

    @classmethod
    def from_random(cls, nside, size, 
                    bands,  
                    mjd_range, skynoise_range,
                    ra_range=None, dec_range=None, **kwargs):
        """ 

        Parameters
        ----------
        nside : int
            healpix nside parameter

        size: int
            number of observations to draw

        bands: list of str
            list of bands that should be drawn.

        ra_range, dec_range: 2d-array, None
            min and max to define a coordinate range to be considered.
            None means no limit.
        
        **kwargs goes to the draw_random() method

        Returns
        -------
        class instance
        """
        this = cls(nside=nside)
        this.draw_random(size,  bands,  
                        mjd_range, skynoise_range, 
                        ra_range=ra_range, dec_range=dec_range,
                        inplace=True, **kwargs)
        return this
    
    # ============== #
    #   Methods      #
    # ============== #
    # ------- #
    #  core   #
    # ------- #
    def radec_to_fieldid(self, radec):
        """ get the fieldid of the given (list of) coordinates """
        if type(radec) is pandas.DataFrame:
            ra = np.asarray(radec["ra"].values, dtype="float")
            dec = np.asarray(radec["dec"].values, dtype="float")
        else:
            ra, dec = np.asarray(radec)
            
        return hp.ang2pix(self.nside, (90 - ra) * np.pi/180, dec * np.pi/180)

    # ------- #
    #  draw   #
    # ------- #    
    def draw_random(self, size, 
                    bands, mjd_range, skynoise_range,
                    gain_range=1, zp_range=25,
                    ra_range=None, dec_range=None,
                    inplace=False, nside=None, **kwargs):
        """ draw observations 

        Parameters
        ----------

        size: int
            number of observations to draw

        bands: list of str
            list of bands that should be drawn.

        ra_range, dec_range: 2d-array, None
            min and max to define a coordinate range to be considered.
            None means no limit.

        skynoise_range, gain_range, zp_range: 2d-array, float, int
            range to be considered.
            If float or int, this value will always be used.
            otherwise, uniform distribution between the range assumed.

        inplace: bool
            shall this method replace the current self.data or
            return a new instance of the class with the 
            generated observing data.
            
        nside: int
            = ignore if inplace is set to True =
            provide a new healpix nside parameters.

        Returns
        -------
        class instance or None
            see the inplace option.

        See also
        --------
        from_random: generate random observing data and loads the instance.
        set_data: set the observing data to the instance.
        """
        if nside is None: # don't change nside
            nside = self.nside
            
        elif inplace: # change nside
            warnings.warn("Cannot change nside with inplace=True, a copy (inplace=False) is returned.")
            inplace = False
            
        data = self._draw_random(nside, size, 
                                 bands, mjd_range, skynoise_range, 
                                 ra_range=ra_range, dec_range=dec_range,
                                 gain_range=gain_range, zp_range=zp_range,
                                 **kwargs)
        
        if not inplace:
            return self.__class__.from_data(nside=nside, data=data)

        self.set_data(data)
        
    # ----------- #
    #  PLOTTER    #
    # ----------- #
    def show(self, stat='size', column=None, title=None, data=None, **kwargs):
        """ shows the sky coverage using ``healpy.mollview`` 

        Parameters
        ----------
        stat: str
            element to be passed to groupby.agg() 
            could be e.g.: 'mean', 'std' etc.
            If stat = 'size', this returns data["fieldid"].value_counts()
            (slightly faster than groupby("fieldid").size()).

        columns: str
            column of the dataframe the stat should be applied to.
            = ignored if stat='size' = 

        title: str
            title of the healpy.mollview plot.
            (healpy.mollview option)
        
        data: pandas.DataFrame, None
            data you want this to be applied to.
            if None, a copy of self.data is used.
            = leave to None if unsure =
            
        Returns
        -------
        None
        
        See also
        --------
        get_fieldstat: get observing statistics for the fields
        """
        data = self.get_fieldstat(stat=stat, columns=column, incl_zeros=True, fillna=np.NaN, data=data)
        return hp.mollview(data, title=title, **kwargs)
        
    # ============== #
    # Static Methods #
    # ============== #        
    @staticmethod
    def _draw_random(nside, size, 
                     bands,  
                     mjd_range, skynoise_range,
                     gain_range=1,
                     zp_range=[27,30],
                     ra_range=None, dec_range=None):
        """ draw observations = internal =

        Parameters
        ----------
        nside : int
            healpix nside parameter

        size: int
            number of observations to draw

        bands: list of str
            list of bands that should be drawn.

        ra_range, dec_range: 2d-array, None
            min and max to define a coordinate range to be considered.
            None means no limit.

        skynoise_range, gain_range, zp_range: 2d-array, float, int
            range to be considered.
            If float or int, this value will always be used.
            otherwise, uniform distribution between the range assumed.

        inplace: bool
            shall this method replace the current self.data or
            return a new instance of the class with the 
            generated observing data.
            
        nside: int
            = ignore if inplace is set to True =
            provide a new healpix nside parameters.

        Returns
        -------
        class instance or None
            see the inplace option.

        See also
        --------
        from_random: generate random observing data and loads the instance.
        draw_random: main function calling _draw_random
        """
        # np.resize(1, 2) -> [1,1]
        mjd = np.random.uniform(*np.resize(mjd_range,2), size=size)
        band = np.random.choice(bands, size=size)
        skynoise = np.random.uniform(*np.resize(skynoise_range, 2), size=size)
        gain = np.random.uniform(*np.resize(gain_range, 2), size=size)
        zp = np.random.uniform(*np.resize(zp_range, 2), size=size)
        # = coords
        # no radec limit
        if ra_range is None and dec_range is None:
            npix = hp.nside2npix(nside)
            ipix = np.random.uniform(0, npix, size=size)
        else:
            ipix_ok = get_ipix_in_range(nside, ra_range=ra_range, dec_range=dec_range)
            ipix = np.random.choice(ipix_ok, size=size)
            
        # data sorted by mjd
        data = pandas.DataFrame(zip(mjd, band, skynoise, gain, zp, ipix),
                               columns=["mjd","band","skynoise", "gain", "zp","fieldid"]
                               ).sort_values("mjd"
                               ).reset_index(drop=False) # don't need to know the creation order
        return data
    
    # ============== #
    #   Properties   #
    # ============== #
    @property
    def nside(self):
        """ healpix nside parameter (defines the 'fields' size and number) """
        return self._nside
    
    @property
    def nfields(self):
        """ number of fields (shortcut to npix) """
        return self.npix
    
    @property    
    def npix(self):
        """ number of healpix pixels """
        if not hasattr(self, "_npix") or self._npix is None:
            self._npix = hp.nside2npix(self.nside)
            
        return self._npix
    
    
    @property
    def metadata(self):
        """ pandas Series containing meta data i formation """
        meta = super().metadata
        meta["nside"] = self.nside
        return meta
