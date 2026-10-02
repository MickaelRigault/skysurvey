"""Utilities and model for Milky Way dust extinction."""

from astropy.coordinates import SkyCoord


def get_mwebv(ra, dec, which="planck"):
    """Get the Milky Way E(B-V) extinction parameter for input coordinates.

    This is based on `dustmaps`. If this is the first time you use it, you may
    have to download the maps first (instructions will be given).

    Parameters
    ----------
    ra, dec : float or array_like
        Coordinates in degrees.

    which : {'planck', 'sfd'}, optional
        Name of the dustmap to use:

        - 'planck': Planck (2013)
        - 'sfd': Schlegel, Finkbeiner & Davis (1998)

        The default is 'planck'.

    Returns
    -------
    float or numpy.ndarray
        E(B-V) values.

    Raises
    ------
    NotImplementedError
        If `which` is neither 'planck' nor 'sfd'.
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
