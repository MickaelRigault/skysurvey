from .survey import GridSurvey
from ztffields.fields import Fields

__all__ = ["ZTF"]

class ZTF( GridSurvey ):

    
    def __init__(self, data=None, level="quadrant", **kwargs):
        """ """
        
        footprint = Fields.get_contours(level=level,
                                              as_polygon=True, allow_multipolygon=True)
        fields = Fields.get_field_geometry(level=level)
        
        super().__init__(data=data, fields=fields, footprint=footprint)
        self._level = level
        
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

    def show_ztf(self, data=None, fieldstat=None, **kwargs):
        """ shows the sky coverage 

        data: pandas.DataFrame
            data to be consider to get the field statistics.
            fieldstat will be derived from that (main grid only) groupby(fieldid).size()
            = ignored is fieldstat is given = 
            
        fieldstat: pandas.Series
            field statistics.

        **kwargs goes to ztffields.skyplot_fields
        """
        import ztffields
        if fieldstat is None:
            if data is None:
                data = self.data
                
            datamain = data[data["fieldid"]<1000] # main grid
            fieldstat = datamain.groupby("expid").first().groupby("fieldid").size()

        fig = ztffields.skyplot_fields(fieldstat, 
                                        label="number of observations (main grid)",
                                           **kwargs)
        return fig

