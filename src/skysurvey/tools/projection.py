"""
This module provides spatial utility functions for projecting camera footprints onto the sky and matching coordinates to survey fields.
"""

import warnings
import numpy as np
import pandas
import geopandas

import shapely
from shapely import geometry

_DEG2RA = np.pi / 180 # compute once.


def skyarea_to_skysurface(skyarea, frac=True, in_deg2=True, projection_correction=True):
    """
    Compute the sky surface covered by a (RA, Dec) geometry.

    By default, the geometry is projected onto the cylindrical equal-area plane
    (RA in radian, sin(Dec)) such that its planar area is the true solid angle.
    Overlapping parts are merged first so they are only counted once.

    Parameters
    ----------
    skyarea : shapely.Geometry or array of shapely.Geometry
        Sky footprint with coordinates (RA, Dec) in degrees.

    frac : bool, optional
        If True, return the area as a fraction of the full sky (4π steradians).
        This takes precedence over `in_deg2`. Default is True.

    in_deg2 : bool, optional
        Only used if `frac` is False. If True, return the area in
        square degrees, otherwise in steradians. Default is True.

    projection_correction : bool, optional
        If True, account for the spherical projection by using
        (RA, sin(Dec)) coordinates, giving the true solid angle.
        If False, use plain (RA, Dec) in radian, i.e. the flat-sky area,
        which overestimates the surface away from the equator.
        Default is True.

    Returns
    -------
    float
        The sky surface as a fraction of the full sky (`frac=True`),
        in square degrees (`in_deg2=True`) or in steradians.

    Notes
    -----
    Only the vertices are projected: edges remain straight lines in the
    projected plane. Use `shapely.segmentize` beforehand for accurate results
    with large polygons. RA wrapping at 0/360 is not handled.

    Examples
    --------
    >>> import shapely
    >>> skyarea = shapely.box(0, -10, 20, 10)
    >>> skyarea_to_skysurface(skyarea, frac=False, in_deg2=True)  # ~398 deg2
    >>> skyarea_to_skysurface(skyarea, frac=False, in_deg2=True, projection_correction=False)  # 400deg2
    """
    def apply_sinprojection(geom):
        """ (ra, dec) in deg -> (ra in rad, sin(dec)) """
        def transform_to_sinprojection(coords):  # coords: (N, 2) array
            ra, dec = np.deg2rad(coords).T
            if projection_correction:
                return np.column_stack([ra, np.sin(dec)])
            return np.column_stack([ra, dec])

        return shapely.transform(geom, transform_to_sinprojection)


    surfacearea = shapely.unary_union( apply_sinprojection(skyarea) ).area
    if frac:
        return surfacearea / (4*np.pi) # full sky is 4*pi
    if in_deg2:
        return surfacearea * (180/np.pi)**2 # in deg2
    return surfacearea # in steradian

def radecmodel_to_skysurface(radecmodel, favor_skyarea=True,
                            ntrial=2e5, frac=True):
    """
    Compute the sky area covered by points drawn from a `ModelDAG` model in RA/Dec space.

    This function samples points from a `ModelDAG` model, projects them onto a unit sphere,
    and computes the convex hull of the projected points to estimate the sky area.
    The area can be returned as a fraction of the total sky (4π steradians) or in steradians.

    Parameters
    ----------
    radecmodel : object
        A `ModelDAG`-compatible model that generates RA/Dec points.

    favor_skyarea : bool, optional
        If True and `radecmodel` has a "skyarea" entry in its "kwargs",
        the area is computed analytically from that geometry using
        :func:`skyarea_to_skysurface` instead of sampling. Default is True.

    ntrial : int, optional
        Number of points to sample from the model. Default is 2e5.
        Tests suggest 2e5 is good at 0.01%.

    frac : bool, optional
        If True, return the area as a fraction of the total sky (4π steradians).
        If False, return the area in steradians. Default is True.

    Returns
    -------
    float
        The sky area covered by the sampled points. If `frac=True`, the value is a fraction of the total sky.
        If `frac=False`, the value is in steradians.

    Notes
    -----
    - The RA/Dec points are converted to radians and projected using `sin(dec)` to account for spherical geometry.
    - The convex hull of the projected points is computed to estimate the sky area.
    - The total sky area is 4π steradians, which corresponds to 41253 square degrees.

    Examples
    --------
    >>> from modeldag import ModelDAG
    >>> radecmodel = ...  # Your ModelDAG-compatible RA/Dec model
    >>> area_frac = radecmodel_to_skysurface(radecmodel, ntrial=1e5, frac=True)
    >>> print(f"Fraction of sky covered: {area_frac:.4f}")
    """
    if ((skyarea := radecmodel.get("kwargs",{}).get("skyarea", None)) is not None) and favor_skyarea:
        return skyarea_to_skysurface(skyarea, frac=frac, in_deg2=False)
    elif skyarea is not None:
        warnings.warn("skyarea is provided but favor_skyarea is False. This could lead to unstable results.")
        skyarea = shapely.unary_union(skyarea)
        if type(skyarea) is shapely.MultiPolygon:
            raise NotImplementedError("sampling method for multiple disconnected skyarea regions has not been implemented. Set favor_skyarea=True.")

    # no skyarea or testing sampling hence let's sample.
    from modeldag import ModelDAG
    import geopandas as gpd
    mdag = ModelDAG({"radec": radecmodel})
    data = mdag.draw( int(ntrial) )

    # account for projection delta_ra*delta_sindec
    data["sin_dec_rad"] = np.sin(data["dec"]*_DEG2RA)
    data["ra_rad"] = data["ra"]*_DEG2RA

    # get the (projected) skyarea # in radian
    projected_skyarea = gpd.GeoDataFrame(geometry=gpd.points_from_xy(data["ra_rad"],
                                                           data["sin_dec_rad"])
                              ).union_all().convex_hull
    if frac:
        return projected_skyarea.area / (4*np.pi)

    return projected_skyarea.area # steradian

def project_to_radec(verts_or_polygon, ra, dec):
    """
    Project a geometry (or its vertices) to given ra, dec coordinates.

    Parameters
    ----------
    verts_or_polygon: shapely.Polygon or 2d-array
        geometry or vertices representing the camera footprint in the sky
        if vertices, the format is: x, y = vertices

    ra: float or array
        poiting(s) R.A.

    dec: float or array
        poiting(s) declination

    Returns
    -------
    list
        If input are vertices, returns a list of new verticies.
        If input are geometries, returns a list of new geometries.
    """
    if isinstance(verts_or_polygon, geometry.Polygon): # polygon
        as_polygon = True
        fra, fdec = np.asarray(verts_or_polygon.exterior.xy)
    else:
        as_polygon = False
        fra, fdec = np.asarray(verts_or_polygon)

    ra = np.atleast_1d(ra)
    dec = np.atleast_1d(dec)
    ra_, dec_ = np.squeeze(rot_xz_sph((fra/np.cos(fdec*np.pi/180))[:,None],
                                            fdec[:,None],
                                            dec)
                          )
    ra_ += ra
    pointings = np.asarray([ra_, dec_]).T
    if as_polygon:
        return [geometry.Polygon(p) for p in pointings]

    return pointings

def spatialjoin_radec_to_fields(radec, fields,
                                how="inner", predicate="intersects",
                                index_radec="index_radec",
                                allow_dask=True, **kwargs):
    """
    Join the radec coordinates with the fields.

    Parameters
    ----------
    radec: DataFrame or 2d-array
        coordinates of the points.

        - DataFrame: must have the "ra" and "dec" columns.
            This will use the DataFrame's index are data index.
        - 2d array (shape N,2): returned index will be 'range(len(ra))'

    fields : [`geopandas.geoserie`, `geopandas.geodataframe` or  dict]

        Fields contains the fieldid and fields shapes. Several forms are accepted:

        - dict: {fieldid: 2d-array, fieldid: 2d-array ...}
            here, the 2d-array are the field's vertices.
        - geoserie: geopandas.GeoSeries with index as fieldid and geometry as field's vertices.
        - geodataframe: geopandas.GeoDataFrame with the 'fieldid' column and geometry as field's vertices.

    how : str, optional
        Type of join, see `geopandas.sjoin`. Default is "inner".

    predicate : str, optional
        Binary predicate used for the join, see `geopandas.sjoin`.
        Default is "intersects".

    index_radec : str, optional
        Name of the column storing the index of the input `radec`.
        Default is "index_radec".

    allow_dask : bool, optional
        If True and `dask_geopandas` is installed, use it to speed up
        the join when there are more than 30 000 fields. Default is True.

    **kwargs
        Additional keyword arguments passed to `geopandas.sjoin`.

    Returns
    -------
    `GeoDataFrame`
        (`geometry.sjoin` result)
    """
    # -------- #
    #  Coords  #
    # -------- #
    if type(radec) in [np.ndarray, list, tuple]:
        inshape = np.shape(radec)
        if inshape[-1] != 2:
            raise ValueError(f"shape of radec must be (N, 2), {inshape} given.")

        radec = pandas.DataFrame(np.atleast_2d(radec), columns=["ra","dec"])

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
    # This goes linearly as size of fields
    if len(fields)>30_000 and allow_dask:
        try:
            import dask_geopandas
        except ImportError:
            pass # no more warnings, we will deal with it.
        else:
            if isinstance(type(fields.index), pandas.MultiIndex): # not supported
                fields = dask_geopandas.from_geopandas(fields, npartitions=10)
                geopoints = dask_geopandas.from_geopandas(geopoints, npartitions=10)
            else:
                warnings.warn("cannot use dask_geopandas with MultiIndex fields dataframe")

    sjoined = geopoints.sjoin(fields,  how="inner", predicate="intersects", **kwargs)
    if "dask" in str( type(sjoined) ):
        sjoined = sjoined.compute()

    # multi-index
    if type(fields.index) is pandas.MultiIndex:
        sjoined = sjoined.rename({f"index_right{i}":name for i, name in enumerate(fields.index.names)}, axis=1)
    else:
        sjoined = sjoined.rename({"index_right": fields.index.name}, axis=1)

    return sjoined


def parse_fields(fields):
    """
    Read various formats for fields and returns it as a `geodataframe`.

    Parameters
    ----------
    fields : [`geopandas.geoserie`, `geopandas.geodataframe` or  dict]

        Fields contains the fieldid and fields shapes. Several forms are accepted:

        - dict: {fieldid: 2d-array or regions, fieldid: 2d-array or regions ...}
            here, the 2d-array are the field's vertices or a astropy/ds9 regions
        - `geoserie`: `geopandas.GeoSeries` with index as fieldid and geometry as field's vertices.
        - `geodataframe`: `geopandas.GeoDataFrame` with the 'fieldid' column and geometry as field's vertices.

    Returns
    -------
    `GeoDataFrame` (`geometry.sjoin` result)

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

def regions_to_shapely(region):
    """
    Converts astropy Region into a shapely geometry.

    Parameters
    ----------
    region: str or Regions (see astropy-regions.readthedocs.io)

        - If str, it is assumed to be in the dr9 ircs format, e.g.: ``region = "box(40.0, 50.0, 5.0, 4.0, 0.0)"``.
        - If Regions, region will be converted into the str format, using ``region = region.serialize("ds9").strip().split("\\n")[-1]``.

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

    if "regions.shapes" in str(type(region)):
        # Regions format -> dr9 icrs format
        region = region.serialize("ds9").strip().split("\n")[-1]

    tregion = type(region)
    if tregion is not str:
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


#
# Projection coordinates.
#
def cart2sph(vec):
    """
    Converts cartesian [x,y,z] to spherical [r, theta, phi] coordinates
    (in degrees).

    Parameters
    ----------
    vec: array
        x, y, z

    Returns
    -------
    array
        [r, theta, phi]
    """
    x, y ,z = vec
    v = np.sqrt(x**2 + y**2 + z**2)
    return np.asarray([v,
                       (np.arctan2(y,x) / _DEG2RA + 180) % 360 - 180,
                       np.arcsin(z/v) / _DEG2RA])


def sph2cart(vec):
    """
    Converts spherical coordinates [r, theta, phi] to cartesian coordinates [x,y,z].

    Parameters
    ----------
    vec: array
        r, theta, phi ; angles in degrees

    Returns
    -------
    array
        [x, y, z]
    """
    v, l, b = vec[0], np.asarray(vec[1])*_DEG2RA, np.asarray(vec[2])*_DEG2RA # noqa: E741
    return np.asarray([v*np.cos(b)*np.cos(l),
                       v*np.cos(b)*np.sin(l),
                       v*np.sin(b)])

def rot_xz(vec, theta):
    """
    Rotates cartesian vector v [x,y,z] by angle theta around axis (0,1,0).

    Parameters
    ----------
    vec: array
        x, y, z

    theta: float
        angle in degree

    Returns
    -------
    array
        rotated x, y, z
    """
    return [vec[0]*np.cos(theta*_DEG2RA) - vec[2]*np.sin(theta*_DEG2RA),
            vec[1][None,:],
            vec[2]*np.cos(theta*_DEG2RA) + vec[0]*np.sin(theta*_DEG2RA)]

def rot_xz_sph(l, b, theta): # noqa: E741
    """
    Rotate spherical coordinate (l,b = theta, phi) by angle theta around axis (0,1,0)
    (calls to :func:`sph2cart`, :func:`rot_xz` and :func:`cart2sph`).

    Parameters
    ----------
    l, b: float or array
       spherical coordinates, in degrees

    theta: float or array
        rotation angle in degrees

    Returns
    -------
    array
        rotated [theta, phi], in degrees
    """
    v_rot = rot_xz( sph2cart([1,l,b]), theta)
    return cart2sph(v_rot)[1:]
