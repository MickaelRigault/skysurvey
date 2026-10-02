"""SNLS survey class and associated utilities.

This module defines the `SNLS` survey class, including the SNLS field
coordinates, MegaCam footprint, and tools to load the observing logs.
"""

import numpy as np
import pandas
import sncosmo

from .basesurvey import GridSurvey


FIELDID = {'D1': {'ra': 36.450190, 'dec': -4.45065},
           'D2': {'ra': 150.11322, 'dec': +2.21571},
           'D3': {'ra': 214.90738, 'dec': +52.6660},
           'D4': {'ra': 333.89903, 'dec': -17.71961}}

def get_snls_field_coordinates(fieldid_name="fieldid"):
    """Get the radec location of the 4 SNLS fields.

    Parameters
    ----------
    fieldid_name : str, optional
        Name of the fieldid index. The default is 'fieldid'.

    Returns
    -------
    pandas.DataFrame
        Dataframe indexed by field name with 'ra' and 'dec' columns.
    """
    
    data = pandas.DataFrame(FIELDID).T
    data.index.name = fieldid_name
    return data

def get_snls_footprint():
    """Get the SNLS (MegaCam) footprint, a 1-degree side square.

    Returns
    -------
    shapely.geometry.Polygon
        Square footprint centered on (0, 0).
    """
    from shapely import geometry
    footprint = geometry.box(-0.5, -0.5, 0.5, 0.5)
    return footprint

def get_weblogs(url="https://supernovae.in2p3.fr/snls5/snls_obslogs.csv"):
    """Load and parse the SNLS observing logs from the input url.

    Observations are assigned to one of the four SNLS fields (D1 to D4), the
    R.A. and Dec. are converted from radians to degrees and band names are
    lower-cased.

    Parameters
    ----------
    url : str, optional
        URL to the csv logs. The default is
        'https://supernovae.in2p3.fr/snls5/snls_obslogs.csv'.

    Returns
    -------
    pandas.DataFrame
        The parsed observing logs.
    """
    data_snls = pandas.read_csv(url)
    # merge RA, Dec as one of the four fields
    radec_groups = data_snls[["ra","dec"]].round(0).groupby(["ra","dec"]).groups
    fields = pandas.Series(radec_groups, name="index").to_frame()
    fields["fieldid"] = ["D1", "D2", "D3", "D4"]
    # and merge them inside the web-log
    data_snls = data_snls.merge(fields[["index", "fieldid"]].explode("index").set_index("index").sort_index(),
                            left_index=True, right_index=True)
    # url logs provide RA,Dec in radian, skysurvey works in degree
    data_snls[["ra", "dec"]] = data_snls[["ra", "dec"]] * 180 / np.pi
    data_snls["band"] = data_snls["band"].str.lower() # forcing low-cap
    return data_snls

def register_snls_bandpasses(filters=['g', 'r', 'i', 'z', 'y'], prefix="megacampsf", at_radius=13.):
    """Register the SNLS bandpasses to sncosmo assuming a single radius.

    Nothing is done if ``megacampsf::g`` can already be retrieved from
    sncosmo.

    Parameters
    ----------
    filters : list of str, optional
        Names of the SNLS filters. The default is ['g', 'r', 'i', 'z', 'y'].

    prefix : str, optional
        Prefix for the registered MegaCam bandpass names, formatted as
        ``{prefix}::{filter}``. The default is 'megacampsf'.

    at_radius : float, optional
        Radius at which the bandpasses are estimated. The default is 13.
    """
    try:
        sncosmo.get_bandpass("megacampsf::g")
        return
    # not well done...
    except: # noqa
        pass
    
    for filter_ in filters:
        megacamband = sncosmo.get_bandpass(f'megacampsf::{filter_}', at_radius)
        megacamband.name = f'{prefix}::{filter_}'
        sncosmo.register(megacamband, force=True)

    
    
class SNLS( GridSurvey ):
    """A class to model the `SNLS` survey.

    Parameters
    ----------
    data : pandas.DataFrame, optional
        Observing data. The default is None.

    **kwargs
        Passed to ``GridSurvey.__init__``.
    """
    def __init__(self, data=None, **kwargs):
        """Initialize the SNLS class."""
        footprint = get_snls_footprint()
        fields = self._parse_fields(get_snls_field_coordinates(), footprint)
        
        register_snls_bandpasses() # loads bandpass
        super().__init__(data=data, fields=fields, footprint=footprint,
                          **kwargs)

    @classmethod
    def from_logs(cls, logpath=None, **kwargs):
        """Load the survey from the observing logs.

        Parameters
        ----------
        logpath : str, optional
            Path to the logs (as csv); they must contain a 'fieldid' column.
            If None, the official SNLS logs are used:
            https://supernovae.in2p3.fr/snls5/snls_obslogs.csv.
            The default is None.

        **kwargs
            Currently ignored.

        Returns
        -------
        SNLS
            The SNLS survey instance.

        Raises
        ------
        ValueError
            If the input log does not contain a 'fieldid' column.

        See Also
        --------
        from_pointings : Load the survey from observing log data.
        """
        if logpath is None:
            logpath = "https://supernovae.in2p3.fr/snls5/snls_obslogs.csv"
            snls_log = get_weblogs(logpath)
        else:
            snls_log = pandas.read_cvs(logpath)
            if "fieldid" not in snls_log:
                raise ValueError("fieldid is not provided in the input log.")

        return cls.from_pointings(data=snls_log)
    
    @classmethod
    def from_pointings(cls, data, **kwargs):
        """Load the survey from observing log data.

        Parameters
        ----------
        data : pandas.DataFrame or dict
            Observing logs. They must contain the columns: 'zp', 'fieldid',
            'gain', 'skynoise', 'mjd' and 'band'.

        **kwargs
            Passed to ``GridSurvey.__init__``.

        Returns
        -------
        SNLS
            The SNLS survey instance.

        See Also
        --------
        from_logs : Load the data from an input file (or the web).
        """
        if type(data) is dict:
            data = pandas.DataFrame.from_dict(data)
            
        return cls(data=data, **kwargs)
