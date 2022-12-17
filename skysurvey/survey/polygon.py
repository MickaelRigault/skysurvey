""" Survey with Polygon vertices based on Shapely. """

import pandas
import numpy as np
import geopandas
from shapely import geometry

from .core import BaseSurvey
from ztffields.projection import spatialjoin_radec_to_fields, parse_fields


__all__ = ["Survey"] # PolygonSurvey renamed Survey as it is the normal used case.


# ================== #
#                    #
#    Polygon         #
#                    #
# ================== #
class Survey( BaseSurvey ):
    _DEFAULT_FIELDS = None
    
    def __init__(self, data=None, fields=None):
        """ """
        super().__init__(data)
        if fields is None:
            if self._DEFAULT_FIELDS is None:
                raise NotImplementedError("No default fields known for this class. No fields given")
            fields = self._DEFAULT_FIELDS
        
        self._fields = self._parse_fields(fields)
        
    @classmethod
    def from_pointings(cls, data, fields=None):
        """ """
        if type(data) is dict:
            data = pandas.DataFrame.from_dict(data)
            
        return cls(data=data, fields=fields)

        
        
    @classmethod
    def from_random(cls, size, 
                    bands, mjd_range, skynoise_range,
                    fields=None, **kwargs):
        """ 
        fields
        """
        this = cls(fields=fields)
        this.draw_random(size,  bands,  
                        mjd_range, skynoise_range, 
                        inplace=True, **kwargs)
        return this
    
    # ============== #
    #   Methods      #
    # ============== #
    # ------- #
    #  core   #
    # ------- #
    def radec_to_fieldid(self, radec):
        """ get the fieldid associated to the given coordinates 

        Parameters
        ----------
        radec: pandas.DataFrame or 2d array
            
        """
        if type(radec) in [np.ndarray, list, tuple]:
            inshape = np.shape(radec)
            if inshape[-1] != 2:
                raise ValueError(f"shape of radec must be (N, 2), {inshape} given.")
        
            radec = pandas.DataFrame(radec, columns=["ra","dec"])

        
        _keyindex = 'index_radec'
        projection = spatialjoin_radec_to_fields(radec, self.fields,
                                                 index_radec=_keyindex,
                                                 ).sort_values(_keyindex
                                                 ).set_index(_keyindex)[self.fields.index.names]
        return projection
    
    # ------- #
    #  draw   #
    # ------- #        
    def draw_random(self, size, 
                    bands, mjd_range, skynoise_range,
                    gain_range=1, zp_range=25,
                    inplace=False, fieldids=None,
                    **kwargs):
        """ """
        if fieldids is None:
            fieldids = self.fieldids
            
        data = self._draw_random(fieldids, size, bands, mjd_range, skynoise_range, 
                                 gain_range=gain_range, zp_range=zp_range,
                                 **kwargs)
        
        if not inplace:
            return self.__class__.from_data(data=data, fields=self.fields)

        self.set_data(data)
        
    # ----------- #
    #  PLOTTER    #
    # ----------- #
    def show(self, stat='size', column=None, title=None, data=None, **kwargs):
        """ shows the sky coverage """
        raise NotImplementedError("Show function not ready")

    
    # ============== #
    # Static Methods #
    # ============== #
    @staticmethod
    def _parse_fields(fields):
        """ """
        return parse_fields(fields)
    
    @staticmethod
    def _draw_random(fieldids, size, 
                     bands,  
                     mjd_range, skynoise_range,
                     gain_range=1,
                     zp_range=[27,30]):
        """ 
        *_range can be 2d-array [min, max] or single values. 
        """
        # np.resize(1, 2) -> [1,1]
        mjd = np.random.uniform(*np.resize(mjd_range,2), size=size)
        band = np.random.choice(bands, size=size)
        skynoise = np.random.uniform(*np.resize(skynoise_range, 2), size=size)
        gain = np.random.uniform(*np.resize(gain_range, 2), size=size)
        zp = np.random.uniform(*np.resize(zp_range, 2), size=size)
        # = coords
        fieldid = np.random.choice(fieldids, size=size)
        # data sorted by mjd
        data = pandas.DataFrame(zip(mjd, band, skynoise, gain, zp, fieldid),
                               columns=["mjd","band","skynoise", "gain", "zp","fieldid"]
                               ).sort_values("mjd"
                               ).reset_index(drop=False) # don't need to know the creation order
        return data
    
    # ============== #
    #   Properties   #
    # ============== #
    @property
    def fields(self):
        """ geodataframe containing the fields coordinates """
        return self._fields

    @property
    def fieldids(self):
        """ list of fields id """
        return self.fields.index
    
    @property
    def nfields(self):
        """ shortcut to npix """
        return len(self.fields)
        
    @property
    def metadata(self):
        """ """
        # ready to be changed.
        meta = super().metadata
        return meta

