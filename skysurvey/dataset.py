

""" This library concerns the data as observed """


import sncosmo
from astropy.table import Table

def get_obsdata(template, observations, parameters, zp=25, zpsys="ab"):
    """ """
    # observation of that field    
    observations[["zp","zpsys"]] = [zp, zpsys]
    sncosmo_obs = Table.from_pandas(observations.rename({"mjd":"time"}, axis=1)) # sncosmo format
    
    # sn parameters
    list_of_parameters = [p_.to_dict() for i_,p_ in parameters.iterrows()] # sncosmo format
    
    # realize LC
    list_of_observations = sncosmo.realize_lcs(sncosmo_obs,template, list_of_parameters)
    if len(list_of_observations) == 0:
        return None
    
    return pandas.concat([l.to_pandas() for l in list_of_observations],  keys=parameters.index)
