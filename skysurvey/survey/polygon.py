""" Survey with Polygon vertices based on Shapely. """

import pandas
import numpy as np
import geopandas
from shapely import geometry

from .core import Survey



def spatialjoin_radec_to_fields(radec, fields, how="inner", predicate="intersects",
                                    index_radec="index_radec", **kwargs):
    """ 
    radec: DataFrame or 2d-array 
        coordinates of the points. 
        - DataFrame: must have the "ra" and "dec" columns. 
            This will use the DataFrame's index are data index.
        - 2d array (shape N,2): returned index will be 'range(len(ra))'
    
    fields : [geopandas.geoserie, geopandas.geodataframe or  dict]
        fields contains the fieldid and fields shapes. Several forms are accepted:
        - dict: {fieldid: 2d-array, fieldid: 2d-array ...}
            here, the 2d-array are the field's vertices.

        - geoserie: geopandas.GeoSeries with index as fieldid and geometry as field's vertices.
            
        - geodataframe: geopandas.GeoDataFrame with the 'fieldid' column and geometry as field's vertices.

    Returns
    -------
    GeoDataFrame (geometry.sjoin result )
    """
    # -------- #
    #  Coords  #
    # -------- #
    if type(radec) in [np.ndarray, list, tuple]:
        if (inshape:=np.shape(radec))[-1] != 2:
            raise ValueError(f"shape of radec must be (N, 2), {inshape} given.")
        
        radec = pandas.DataFrame(radec, columns=["ra","dec"])

    # Points to be considered
    geoarray = geopandas.points_from_xy(*radec[["ra","dec"]].values.T)
    geopoints = geopandas.GeoDataFrame({index_radec:radec.index}, geometry=geoarray)
    
    # -------- #
    # Fields   #
    # -------- #
    # goes from dict to geoseries (more natural) 
    fields = parse_fields(fields)

    # -------- #
    # Joining  #
    # -------- #
    return geopoints.sjoin(fields,  how="inner", predicate="intersects", **kwargs)


def parse_fields(fields):
    """ read various formats for fields and returns it as a geodataframe

    Parameters
    ----------
    fields : [geopandas.geoserie, geopandas.geodataframe or  dict]
        fields contains the fieldid and fields shapes. Several forms are accepted:
        - dict: {fieldid: 2d-array or regions, fieldid: 2d-array or regions ...}
            here, the 2d-array are the field's vertices or a astropy/ds9 regions

        - geoserie: geopandas.GeoSeries with index as fieldid and geometry as field's vertices.
            
        - geodataframe: geopandas.GeoDataFrame with the 'fieldid' column and geometry as field's vertices.

    Returns
    -------
    GeoDataFrame (geometry.sjoin result)


    Examples
    --------
    provide a dict of ds9 regions
    >>> fields = {450:"box(50,30, 3,4,0)", 541:"ellipse(190,-10,1.5,1,50)"}
    >>> geodf = parse_fields(fields)

    """
    if type(fields) is dict:
        values = fields.values()
        indexes = fields.keys()
        # dict of array goes to shapely.Geometry as expected by geopandas
        test_kind = type( values.__iter__().__next__() ) # check the first
        if test_kind in [np.ndarray, list, tuple]:
            values = [geometry.Polygon(v) for v in values]
            
        if test_kind is str or "regions.shapes" in str(test_kind):
            values = [regions_to_shapely(v) for v in values]
            
        fields = geopandas.GeoSeries(values,  index = indexes)
            
    if type(fields) is geopandas.geoseries.GeoSeries:
        fields = geopandas.GeoDataFrame({"fieldid":fields.index},
                                        geometry=fields.values)
    elif type(fields) is not geopandas.geodataframe.GeoDataFrame:
        raise ValueError("cannot parse the format of the input 'fields' variable. Should be dict, GeoSeries or GeoPandas")

    return fields


# ================= # 
#                   #
#  Astropy Regions  #
#                   #
# ================= #
def regions_to_shapely(region):
    """ 
    Parameters
    ----------
    region: str or Regions (see astropy-regions.readthedocs.io)
        if str, it is assumed to be the dr9 ircs format 
        e.g. region = box(40.00000000,50.00000000,5.00000000,4.00000000,0.00000000)
        if Regions, region will be converted into the str format
        using ``region = region.serialize("ds9").strip().split("\n")[-1]``
        The following format have been implemented:
        - box
        - circle
        - ellipse
        - polygon
        
    Returns
    -------
    Shapely's Geometry
        the geometry will depend on the input regions.
        
    Raises
    ------
    NotImplementedError
        if the format is not recognised.
        
    Examples
    --------
    >>> shapely_ellipse = regions_to_shapely('ellipse(54,43.4, 4, 2,-10)')
    >>> shapely_rotated_rectangle = regions_to_shapely('box(-30,0.4, 4, 2,80)')
    """
    import shapely
    from shapely import geometry
    
    if "regions.shapes" in str(type(region)):
        # Regions format -> dr9 icrs format
        region = region.serialize("ds9").strip().split("\n")[-1]
        
    if (tregion:=type(region)) is not str:
        raise ValueError(f"cannot parse the input region format ; {tregion} given")
        
    # it works, let's parse it.
    which, params = region.replace(")","").split("(")
    params = np.asarray(params.split(","), dtype="float")
    
    # Box, 
    if which == "box": # rectangle
        centerx, centery, width, height, angle = params
        minx, miny, maxx, maxy = centerx-width, centery-height, centerx+width, centery+height
        geom = geometry.box(minx, miny, maxx, maxy, ccw=True)
        if angle != 0:
            geom = shapely.affinity.rotate(geom, angle)
            
    # Cercle        
    elif which == "circle":
        centerx, centery, radius = params
        geom = geometry.Point(centerx, centery).buffer(radius)
        
    # Ellipse
    elif which == "ellipse":
        centerx, centery, a, b, theta = params
        # unity circle
        geom = geometry.Point(centerx, centery).buffer(1)
        geom = shapely.affinity.scale(geom, a,b)
        if theta != 0:
            geom = shapely.affinity.rotate(geom, theta)
        
    # Ellipse        
    elif which == "polygon":
        params = (params + 180) %360 - 180
        coords = params.reshape(int(len(params)/2),2)
        geom = geometry.Polygon(coords)
        
    else:
        raise NotImplementedError(f"the {which} form not implemented. box, circle, ellpse and polygon are.")
    
    # shapely's geometry
    return geom


# ================== #
#                    #
#    Polygon         #
#                    #
# ================== #
class PolygonSurvey( Survey ):
    _DEFAULT_FIELDS = None
    
    def __init__(self, data=None, fields=None):
        """ """
        super().__init__(data)
        if fields is None:
            if self._DEFAULT_FIELDS is None:
                raise NotImplementedError("No default fields known for this class. No fields given")
            fields = self._DEFAULT_FIELDS
        
        self._fields = self._parse_fields(fields)
        
    @classmethod
    def from_pointings(cls, data, fields=None):
        """ """
        if type(data) is dict:
            data = pandas.DataFrame.from_dict(data)
            
        return cls(data=data, fields=fields)

        
        
    @classmethod
    def from_random(cls, size, 
                    bands, mjd_range, skynoise_range,
                    fields=None, **kwargs):
        """ 
        fields
        """
        this = cls(fields=fields)
        this.draw_random(size,  bands,  
                        mjd_range, skynoise_range, 
                        inplace=True, **kwargs)
        return this
    
    # ============== #
    #   Methods      #
    # ============== #
    # ------- #
    #  core   #
    # ------- #
    def radec_to_fieldid(self, ra, dec):
        """ get the fieldid associated to the given coordinates """
        return spatialjoin_radec_to_fields(pandas.DataFrame({"ra":ra,"dec":dec}),
                                           self.fields, index_radec="index_radec"
                                           ).groupby("index_radec")["fieldid"].apply(list)

    # ------- #
    #  draw   #
    # ------- #        
    def draw_random(self, size, 
                    bands, mjd_range, skynoise_range,
                    gain_range=1, zp_range=25,
                    inplace=False, fieldids=None,
                    **kwargs):
        """ """
        if fieldids is None:
            fieldids = self.fieldids
            
        data = self._draw_random(fieldids, size, bands, mjd_range, skynoise_range, 
                                 gain_range=gain_range, zp_range=zp_range,
                                 **kwargs)
        
        if not inplace:
            return self.__class__.from_data(data=data, fields=self.fields)

        self.set_data(data)
        
    # ----------- #
    #  PLOTTER    #
    # ----------- #
    def show(self, stat='size', column=None, title=None, data=None, **kwargs):
        """ shows the sky coverage """
        raise NotImplementedError("Show function not ready")
        
    # ============== #
    # Static Methods #
    # ============== #
    @staticmethod
    def _parse_fields(fields):
        """ """
        return parse_fields(fields)
    
    @staticmethod
    def _draw_random(fieldids, size, 
                     bands,  
                     mjd_range, skynoise_range,
                     gain_range=1,
                     zp_range=[27,30]):
        """ 
        *_range can be 2d-array [min, max] or single values. 
        """
        # np.resize(1, 2) -> [1,1]
        mjd = np.random.uniform(*np.resize(mjd_range,2), size=size)
        band = np.random.choice(bands, size=size)
        skynoise = np.random.uniform(*np.resize(skynoise_range, 2), size=size)
        gain = np.random.uniform(*np.resize(gain_range, 2), size=size)
        zp = np.random.uniform(*np.resize(zp_range, 2), size=size)
        # = coords
        fieldid = np.random.choice(fieldids, size=size)
        # data sorted by mjd
        data = pandas.DataFrame(zip(mjd, band, skynoise, gain, zp, fieldid),
                               columns=["mjd","band","skynoise", "gain", "zp","fieldid"]
                               ).sort_values("mjd"
                               ).reset_index(drop=False) # don't need to know the creation order
        return data
    
    # ============== #
    #   Properties   #
    # ============== #
    @property
    def fields(self):
        """ geodataframe containing the fields coordinates """
        return self._fields

    @property
    def fieldids(self):
        return self.fields["fieldid"].values
    
    @property
    def nfields(self):
        """ shortcut to npix """
        return len(self.fields)
        
    @property
    def metadata(self):
        """ """
        # ready to be changed.
        meta = super().metadata
        return meta
