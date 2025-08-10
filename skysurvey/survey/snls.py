import numpy as np
import pandas
import sncosmo

from .basesurvey import GridSurvey

__all__ = ["SNLS"]



FIELDID = {'D1': {'ra': 36.450190, 'dec': -4.45065},
           'D2': {'ra': 150.11322, 'dec': +2.21571},
           'D3': {'ra': 214.90738, 'dec': +52.6660},
           'D4': {'ra': 333.89903, 'dec': -17.71961}}

def get_snls_field_coordinates(fieldid_name="fieldid"):
    """ get the radec location of the 4 SNLS fields

    Parameters
    ----------
    fieldid_name: str
        name of the fieldid column.

    Returns
    -------
    pandas.DataFrame
    """
    
    data = pandas.DataFrame(FIELDID).T
    data.index.name = fieldid_name
    return data

def get_snls_footprint():
    """ returns a 1-degree side square footprint

    Returns
    -------
    shapely.geometry.Polygon
    """
    from shapely import geometry
    footprint = geometry.box(-0.5, -0.5, 0.5, 0.5)
    return footprint

def get_weblogs(url="https://supernovae.in2p3.fr/snls5/snls_obslogs.csv"):
    """ load and parse data from the input url 
    
    Parameters
    ----------
    url: str
        url to the data.

    Returns
    -------
    pandas.DataFrame
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
    """ register snls band passes to sncosmo assuming a single radius 

    Parameters
    ----------
    filters: list
        names of the snls filters

    prefix: str
        prefix for the megacam filters format: {prefix}{filter}

    at_radius: float
        bandpass will be estimated at this entry radius.

    Returns
    -------
    None
    """
    try:
        sncosmo.get_bandpass("megacampsf::g")
        return
    except: # not well done...
        pass
    
    for filter_ in filters:
        megacamband = sncosmo.get_bandpass(f'megacampsf::{filter_}', at_radius)
        megacamband.name = f'{prefix}::{filter_}'
        sncosmo.register(megacamband, force=True)

    
    
class SNLS( GridSurvey ):
    
    def __init__(self, data=None, **kwargs):
        """ 
        Initialize the SNLS class.

        Parameters
        ----------
        data: pandas.DataFrame
            observing data.

        **kwargs goes to GridSurvey.__init__
        """
        footprint = get_snls_footprint()
        fields = self._parse_fields(get_snls_field_coordinates(), footprint)
        
        register_snls_bandpasses() # loads bandpass
        super().__init__(data=data, fields=fields, footprint=footprint,
                          **kwargs)

    @classmethod
    def from_logs(cls, logpath=None, **kwargs):
        """ loads the data from the observing logs. 

        If None provided, this uses:
        https://supernovae.in2p3.fr/snls5/snls_obslogs.csv

        Parameters
        ----------
        logpath: path
            filepath to where the logs are stored (as csv).
            If None, the official snls webpage is used:
            https://supernovae.in2p3.fr/snls5/snls_obslogs.csv
            
        **kwargs goes to GridSurvey.__init__

        Returns
        -------
        instance
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
        """ loads from observing log data 

        Parameters
        ----------
        data: pandas.DataFrame, dict
            observing logs, must contains: 
            ['zp', 'fieldid', 'gain', 'skynoise', 'mjd', 'band']

        **kwargs goes to GridSurvey.__init__

        Returns
        -------
        instance

        See also:
        ---------
        from_logs(): loads the data from input file (or web).
        """
        if type(data) is dict:
            data = pandas.DataFrame.from_dict(data)
            
        return cls(data=data, **kwargs)

            
        
