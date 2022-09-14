from .polygon import PolygonSurvey
from ztfquery.fields import get_field_vertices

ZTF_FIELDS = get_field_vertices(as_dict=True)

__all__ = ["ZTF"]

class ZTF( PolygonSurvey ):
    _DEFAULT_FIELDS  = ZTF_FIELDS
    
    @classmethod
    def from_realistic(nyears=4):
        raise NotImplementedError("from_realistic not implemented.")
