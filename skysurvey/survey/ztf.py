from .polygon import Survey
from ztffields.fields import Fields

__all__ = ["ZTF"]

class ZTF( Survey ):

    def __init__(self, data=None, level="focalplane", **kwargs):
        """ """
        fields = Fields.get_field_geometry(level=level)
        return super().__init__(data=data, fields=fields)
    
    @classmethod
    def from_pointings(cls, data, level="quadrant"):
        """ """
        if type(data) is dict:
            data = pandas.DataFrame.from_dict(data)
            
        return cls(data=data, level=level)


    @classmethod
    def from_random(cls, size, bands, mjd_range, skynoise_range,
                    level="focalplane", **kwargs):
        """ 
        fields
        """
        this = cls(level=level)
        this.draw_random(size, bands,  
                        mjd_range, skynoise_range, 
                        inplace=True, **kwargs)
        return this
