"""ZTF survey class and associated field geometry.

This module defines the `ZTF` survey class, including ZTF field geometry at
different levels (quadrant, CCD, field) and tools to load the observing logs.
"""

import pandas

from .basesurvey import GridSurvey
from ztffields.fields import Fields


class ZTF( GridSurvey ):
    """A class to model the `ZTF` survey.

    Parameters
    ----------
    data : pandas.DataFrame, optional
        Observing data. The default is None.

    level : {'quadrant', 'ccd', 'field'}, optional
        Level of the ZTF fields. The default is 'quadrant'.

    **kwargs
        Currently ignored.
    """
    def __init__(self, data=None, level="quadrant", **kwargs):
        """Initialize the ZTF class."""
        
        footprint = Fields.get_contours(level=level,
                                        as_polygon=True,
                                        allow_multipolygon=True)
        fields = Fields.get_field_geometry(level=level)
        
        super().__init__(data=data, fields=fields, footprint=footprint)
        self._level = level
        
    @classmethod
    def from_logs(cls, **kwargs):
        """Load the ZTF survey from the observing logs.

        The logs are obtained from ``ztfcosmo.get_observing_logs()`` (requires
        the `ztfcosmo` package) and loaded at the 'quadrant' level.

        Parameters
        ----------
        **kwargs
            Currently ignored.

        Returns
        -------
        ZTF
            The ZTF survey instance.

        Raises
        ------
        ImportError
            If `ztfcosmo` is not installed.
        """
        try:
            import ztfcosmo
        except ImportError:
            raise ImportError("you need to install ztfcosmo => pip install ztfcosmo")
        
        logs = ztfcosmo.get_observing_logs()
        return cls.from_pointings(data=logs, level="quadrant")
        
    @classmethod
    def from_pointings(cls, data, level="quadrant"):
        """Load the ZTF survey from pointings.

        Parameters
        ----------
        data : pandas.DataFrame or dict
            Observing data.

        level : {'quadrant', 'ccd', 'field'}, optional
            Level of the ZTF fields. The default is 'quadrant'.

        Returns
        -------
        ZTF
            The ZTF survey instance.
        """
        if type(data) is dict:
            data = pandas.DataFrame.from_dict(data)
            
        return cls(data=data, level=level)

    def get_skyarea(self, observed=True, buffer=0.5):
        """Compute the total sky area covered by the survey fields.

        Parameters
        ----------
        observed : bool, optional
            If True, only fields present in the observation log are included.
            If False, the area is calculated using all fields defined in the
            survey. The default is True.

        buffer : float, optional
            Size of the padding (in degrees) to apply around the combined
            geometry. This helps smooth overlaps and fill small gaps between
            neighboring fields. The default is 0.5.

        Returns
        -------
        shapely.geometry.Polygon or shapely.geometry.MultiPolygon
            The combined sky coverage.
        """
        import shapely
        list_of_geoms = self.fields["geometry"]
        if observed:
            list_of_geoms = list_of_geoms.loc[ self.data["fieldid"].unique() ]
            
        return shapely.unary_union(list_of_geoms).buffer(buffer)


    def show(self, *args, **kwargs):
        """Shortcut to :meth:`show_ztf`.

        Parameters
        ----------
        *args
            Passed to :meth:`show_ztf`.

        **kwargs
            Passed to :meth:`show_ztf`.

        Returns
        -------
        matplotlib.figure.Figure
            The sky coverage figure.
        """
        return self.show_ztf(*args, **kwargs)
    
    def show_ztf(self, data=None, fieldstat=None, **kwargs):
        """Show the sky coverage.

        Parameters
        ----------
        data : pandas.DataFrame, optional
            Data used to derive the field statistics: number of exposures per
            field of the main grid (fieldid < 1000). Ignored if `fieldstat` is
            given. If None, ``self.data`` is used.

        fieldstat : pandas.Series, optional
            Field statistics (value per fieldid). If None, it is derived from
            `data`.

        **kwargs
            Passed to ``ztffields.skyplot_fields``.

        Returns
        -------
        matplotlib.figure.Figure
            The sky coverage figure.
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

