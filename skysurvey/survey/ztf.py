from .polygon import Survey
from ztffields.fields import Fields

__all__ = ["ZTF"]

class ZTF( Survey ):

    def __init__(self, data=None, level="focalplane", **kwargs):
        """ """
        fields = Fields.get_field_geometry(level=level)
        return super().__init__(data=data, fields=fields)

    @classmethod
    def from_logs(cls, **kwargs):
        """ """
        from ztfquery.skyvision import get_summary_logs
        logs = get_summary_logs(**kwargs)
        return cls.from_pointings(data=logs, level="quadrant")
        
        
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


    def show(self, data=None, **kwargs):
        """ shows the sky coverage 

        **kwargs goes to ztffields.skyplot_fields
        """
        import ztffields
        if data is None:
            data = self.data
        
        datamain = data[data["fieldid"]<1000] # main grid
        fieldid_s = datamain.groupby("expid").first().groupby("fieldid").size()

        fig = ztffields.skyplot_fields(fieldid_s, 
                                        label="number of observations (main grid)",
                                           **kwargs)

