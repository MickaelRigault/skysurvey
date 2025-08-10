
from astropy.coordinates import SkyCoord

__all__ = ["mwebv_model"]

def get_mwebv(ra, dec, which="planck"):
    """ get the mikly way E(B-V) extinction parameter for input coordinates

    This is based on dustmaps. 
    If this is the first time you use it, you may have to download the maps 
    first (instruction will be given)
    
    Parameters
    ----------
    ra, dec: float, array
        coordinates

    which: string
        name of the dustmap to use.
        - planck: Planck 2013
        - SFD: 

    Returns
    -------
    array
        E(B-V) values.
    """
    if which.lower() == "planck":
        from dustmaps.planck import PlanckQuery as dustquery
    elif which.lower() == "sdf":
        from dustmaps.sfd import SFDQuery as dustquery
    else:
        raise NotImplementedError("Only Planck and SDF maps implemented")
        
    coords = SkyCoord(ra, dec, unit="deg")
    return dustquery()(coords) # Instanciate and call.


mwebv_model = {"mwebv": {"func": get_mwebv,
                         "kwargs":{"ra":"@ra", "dec":"@dec"}}
              }
