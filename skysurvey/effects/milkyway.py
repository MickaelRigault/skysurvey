
from astropy.coordinates import SkyCoord

__all__ = ["mwebv_model"]

def get_mwebv(ra, dec, which="planck"):
    """ Get the Milky Way E(B-V) extinction parameter for input coordinates.

    This is based on dustmaps. 
    If this is the first time you use it, you may have to download the maps 
    first (instructions will be given).
    
    Parameters
    ----------
    ra, dec: float, array
        Coordinates.

    which: string
        Name of the dustmap to use.
        - planck: Planck (2013)
        - sfd: Schlegel, Finkbeiner & Davis (1998)

    Returns
    -------
    array
        E(B-V) values.
    """
    if which.lower() == "planck":
        from dustmaps.planck import PlanckQuery as dustquery
    elif which.lower() == "sfd":
        from dustmaps.sfd import SFDQuery as dustquery
    else:
        raise NotImplementedError("Only Planck and SFD maps implemented")
        
    coords = SkyCoord(ra, dec, unit="deg")
    return dustquery()(coords) # Instanciate and call.


mwebv_model = {"mwebv": {"func": get_mwebv,
                         "kwargs":{"ra":"@ra", "dec":"@dec"}}
              }
