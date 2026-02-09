from shapely import geometry
from skysurvey.tools import utils
from skysurvey.survey.polygon import project_to_radec

def test_project_to_radec():
    """ """
    footprint = geometry.Point(0,0).buffer(1)
    ra, dec = utils.random_radec(size=20, ra_range=[200,250], dec_range=[-20,10])
    _ = project_to_radec(footprint, ra, dec)
    assert len(_) == 20
