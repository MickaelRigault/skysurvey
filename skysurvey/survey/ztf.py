from .basesurvey import GridSurvey
from ztffields.fields import Fields

__all__ = ["ZTF"]

class ZTF( GridSurvey ):

    
    def __init__(self, data=None, level="quadrant", **kwargs):
        """ 
        Initialize the ZTF class.

        Parameters
        ----------
        data: pandas.DataFrame
            observing data.

        level: str
            level of the ZTF fields (quadrant, ccd, field).

        **kwargs goes to GridSurvey.__init__
        """
        
        footprint = Fields.get_contours(level=level,
                                        as_polygon=True,
                                        allow_multipolygon=True)
        fields = Fields.get_field_geometry(level=level)
        
        super().__init__(data=data, fields=fields, footprint=footprint)
        self._level = level
        
    @classmethod
    def from_logs(cls, **kwargs):
        """ 
        Load the ZTF survey from the logs.

        **kwargs goes to from_pointings

        Returns
        -------
        ZTF
        """
        try:
            import ztfcosmo
        except:
            raise ImportError("you need to install ztfcosmo => pip install ztfcosmo")
        
        logs = ztfcosmo.get_observing_logs()
        return cls.from_pointings(data=logs, level="quadrant")
        
    @classmethod
    def from_pointings(cls, data, level="quadrant"):
        """ 
        Load the ZTF survey from pointings.

        Parameters
        ----------
        data: pandas.DataFrame or dict
            observing data, must contain the rakey and deckey columns.

        level: str
            level of the ZTF fields (quadrant, ccd, field).

        Returns
        -------
        ZTF
        """
        if type(data) is dict:
            data = pandas.DataFrame.from_dict(data)
            
        return cls(data=data, level=level)

    def show_ztf(self, data=None, fieldstat=None, **kwargs):
        """Show the sky coverage.

        Parameters
        ----------
        data: pandas.DataFrame, optional
            Data to be considered to get the field statistics.
            fieldstat will be derived from that (main grid only) groupby(fieldid).size().
            Ignored is fieldstat is given.
        fieldstat: pandas.Series, optional
            Field statistics.
        **kwargs
            Goes to ztffields.skyplot_fields.

        Returns
        -------
        matplotlib.figure
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

