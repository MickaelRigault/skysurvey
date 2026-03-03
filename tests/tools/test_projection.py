import numpy as np
from shapely import geometry
from skysurvey.tools import utils
from skysurvey.survey.polygon import project_to_radec
from skysurvey.tools.projection import radecmodel_to_skysurface, cart2sph, sph2cart

# tests for the func radecmodel_to_skysurface
def test_radecmodel_to_skysurface():
    """ """
    
    radec = {"func": utils.random_radec,
             "kwargs": {"dec_range":[0, 90], "ra_range":[0, 360]},
             "as": ["ra","dec"]
            }

    farea = radecmodel_to_skysurface(radec, ntrial=1e4, frac=True)
    assert np.isclose(farea, 0.5, rtol=0.1)

    farea = radecmodel_to_skysurface(radec, ntrial=1e4, frac=False)
    assert np.isclose(farea, 2*np.pi, rtol=0.1) # 4pi/2

    # update for full sky
    radec["kwargs"]= {"dec_range":[-90, 90], "ra_range":[0, 360]}
             
    farea = radecmodel_to_skysurface(radec, ntrial=1e4, frac=True)
    assert np.isclose(farea, 1., rtol=0.1) # 4pi/2    

# tests for the func project_to_radec
def test_project_to_radec():
    """ """
    footprint = geometry.Point(0,0).buffer(1)
    ra, dec = utils.random_radec(size=20, ra_range=[200,250], dec_range=[-20,10])
    _ = project_to_radec(footprint, ra, dec)
    assert len(_) == 20



# tests for the func cart2sph
def test_cart2sph():
    vec = np.array([0.4, -0.2, 0.9160254])
    vec /= np.linalg.norm(vec)

    sph = cart2sph(vec)
    cart = sph2cart(sph)

    np.testing.assert_allclose(cart, vec, atol=1e-12)

# tests for the func sph2cart
def test_sph2cart():
    np.testing.assert_allclose(sph2cart([1.,0.,0.]), [1.,0.,0.], atol=1e-12)
    np.testing.assert_allclose(sph2cart([1.,90.,0.]), [0.,1.,0.], atol=1e-12)
    np.testing.assert_allclose(sph2cart([1.,0.,90.]), [0.,0.,1.], atol=1e-12) 