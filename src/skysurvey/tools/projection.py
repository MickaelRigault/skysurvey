"""Spatial utilities to project footprints on the sky and match fields.

Includes functions to project camera footprints onto the sky and to match
coordinates to survey fields.
"""

import warnings
import numpy as np
import pandas
import geopandas

import shapely
from shapely import geometry

_DEG2RA = np.pi / 180 # compute once.


def skyarea_to_skysurface(skyarea, frac=True, in_deg2=True, projection_correction=True):
    """Compute the sky surface covered by a (RA, Dec) geometry.

    By default, the geometry is projected onto the cylindrical equal-area plane
    (RA in radian, sin(Dec)) such that its planar area is the true solid angle.
    Overlapping parts are merged first so they are only counted once.

    Parameters
    ----------
    skyarea : shapely.Geometry or array_like of shapely.Geometry
        Sky footprint with coordinates (RA, Dec) in degrees.

    frac : bool, optional
        If True, return the area as a fraction of the full sky (4π steradians).
        This takes precedence over `in_deg2`. The default is True.

    in_deg2 : bool, optional
        Only used if `frac` is False. If True, return the area in square
        degrees, otherwise in steradians. The default is True.

    projection_correction : bool, optional
        If True, account for the spherical projection by using
        (RA, sin(Dec)) coordinates, giving the true solid angle.
        If False, use plain (RA, Dec) in radian, i.e. the flat-sky area,
        which overestimates the surface away from the equator.
        The default is True.

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
        """Project (ra, dec) in deg to (ra in rad, sin(dec))."""
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
    """Compute the sky area covered by points drawn from a RA/Dec `ModelDAG` model.

    This function samples points from a `ModelDAG` model, projects them onto a
    unit sphere, and computes the convex hull of the projected points to
    estimate the sky area. The area can be returned as a fraction of the total
    sky (4π steradians) or in steradians.

    Parameters
    ----------
    radecmodel : dict
        A `ModelDAG`-compatible model entry that generates RA/Dec points.

    favor_skyarea : bool, optional
        If True and `radecmodel` has a "skyarea" entry in its "kwargs",
        the area is computed analytically from that geometry using
        :func:`skyarea_to_skysurface` instead of sampling.
        The default is True.

    ntrial : int or float, optional
        Number of points to sample from the model. Tests suggest 2e5 is good
        at 0.01%. The default is 2e5.

    frac : bool, optional
        If True, return the area as a fraction of the total sky
        (4π steradians). If False, return the area in steradians.
        The default is True.

    Returns
    -------
    float
        The sky area covered by the sampled points. If `frac=True`, the value
        is a fraction of the total sky. If `frac=False`, the value is in
        steradians.

    Raises
    ------
    NotImplementedError
        If sampling is requested (`favor_skyarea=False`) while the skyarea is
        made of multiple disconnected regions.

    Warns
    -----
    UserWarning
        If a skyarea is provided but `favor_skyarea` is False.

    Notes
    -----
    - The RA/Dec points are converted to radians and projected using
      `sin(dec)` to account for spherical geometry.
    - The convex hull of the projected points is computed to estimate the sky
      area.
    - The total sky area is 4π steradians, which corresponds to 41253 square
      degrees.

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
    """Project a geometry (or its vertices) to given RA, Dec coordinates.

    Parameters
    ----------
    verts_or_polygon : shapely.geometry.Polygon or array_like
        Geometry or vertices representing the camera footprint in the sky.
        If vertices, the format is: ``x, y = vertices``.

    ra : float or array_like
        Pointing(s) right ascension, in degrees.

    dec : float or array_like
        Pointing(s) declination, in degrees.

    Returns
    -------
    list of shapely.geometry.Polygon or numpy.ndarray
        If input are vertices, returns an array of new vertices (one per
        pointing). If input is a geometry, returns a list of new geometries.
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
    """Join the RA, Dec coordinates with the fields.

    Parameters
    ----------
    radec : pandas.DataFrame or array_like
        Coordinates of the points.

        - DataFrame: must have the "ra" and "dec" columns. The DataFrame's
          index is used as data index.
        - 2d array (shape N, 2): returned index will be ``range(len(ra))``.

    fields : geopandas.GeoSeries, geopandas.GeoDataFrame, or dict
        Fields containing the fieldid and field shapes. Several forms are
        accepted:

        - dict: {fieldid: 2d-array, fieldid: 2d-array ...}, where the
          2d-arrays are the field's vertices.
        - GeoSeries: index as fieldid and geometry as field's vertices.
        - GeoDataFrame: with the 'fieldid' column and geometry as field's
          vertices.

        See :func:`parse_fields`.

    how : str, optional
        Type of join, see :func:`geopandas.sjoin`. Currently not forwarded:
        an "inner" join is always used. The default is "inner".

    predicate : str, optional
        Binary predicate used for the join, see :func:`geopandas.sjoin`.
        Currently not forwarded: "intersects" is always used.
        The default is "intersects".

    index_radec : str, optional
        Name of the column storing the index of the input `radec`.
        The default is "index_radec".

    allow_dask : bool, optional
        If True and `dask_geopandas` is installed, use it to speed up
        the join when there are more than 30 000 fields. The default is True.

    **kwargs
        Passed to :func:`geopandas.sjoin`.

    Returns
    -------
    geopandas.GeoDataFrame
        Result of the spatial join (:func:`geopandas.sjoin`).

    Raises
    ------
    ValueError
        If `radec` is an array whose shape is not (N, 2).
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
    """Read various formats for fields and return them as a GeoDataFrame.

    Parameters
    ----------
    fields : geopandas.GeoSeries, geopandas.GeoDataFrame, or dict
        Fields containing the fieldid and field shapes. Several forms are
        accepted:

        - dict: {fieldid: 2d-array or region, fieldid: 2d-array or region ...},
          where the 2d-arrays are the field's vertices; regions are
          astropy/ds9 regions (see :func:`regions_to_shapely`).
        - GeoSeries: index as fieldid and geometry as field's vertices.
        - GeoDataFrame: with the 'fieldid' column and geometry as field's
          vertices.

    Returns
    -------
    geopandas.GeoDataFrame
        Fields with a 'fieldid' column and their geometry.

    Raises
    ------
    ValueError
        If the format of `fields` cannot be parsed.

    Examples
    --------
    Provide a dict of ds9 regions:

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
    r"""Convert an astropy Region into a shapely geometry.

    Parameters
    ----------
    region : str or regions.Region
        Region to convert (see astropy-regions.readthedocs.io).

        - If str, it is assumed to be in the ds9 icrs format, e.g.:
          ``region = "box(40.0, 50.0, 5.0, 4.0, 0.0)"``.
        - If Region, it will be converted into the str format, using
          ``region = region.serialize("ds9").strip().split("\n")[-1]``.

        The following formats have been implemented:

        - box
        - circle
        - ellipse
        - polygon

    Returns
    -------
    shapely.Geometry
        The geometry; its type depends on the input region.

    Raises
    ------
    ValueError
        If the input region is not a str (after conversion).

    NotImplementedError
        If the region shape is not recognised.

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
    """Convert cartesian [x, y, z] to spherical [r, theta, phi] coordinates.

    Angles are returned in degrees.

    Parameters
    ----------
    vec : array_like
        Cartesian coordinates x, y, z.

    Returns
    -------
    numpy.ndarray
        Spherical coordinates [r, theta, phi], angles in degrees.
    """
    x, y ,z = vec
    v = np.sqrt(x**2 + y**2 + z**2)
    return np.asarray([v,
                       (np.arctan2(y,x) / _DEG2RA + 180) % 360 - 180,
                       np.arcsin(z/v) / _DEG2RA])


def sph2cart(vec):
    """Convert spherical [r, theta, phi] to cartesian [x, y, z] coordinates.

    Parameters
    ----------
    vec : array_like
        Spherical coordinates r, theta, phi; angles in degrees.

    Returns
    -------
    numpy.ndarray
        Cartesian coordinates [x, y, z].
    """
    v, l, b = vec[0], np.asarray(vec[1])*_DEG2RA, np.asarray(vec[2])*_DEG2RA # noqa: E741
    return np.asarray([v*np.cos(b)*np.cos(l),
                       v*np.cos(b)*np.sin(l),
                       v*np.sin(b)])

def rot_xz(vec, theta):
    """Rotate cartesian vector [x, y, z] by angle theta around axis (0, 1, 0).

    Parameters
    ----------
    vec : array_like
        Cartesian coordinates x, y, z.

    theta : float or array_like
        Rotation angle, in degrees.

    Returns
    -------
    list
        Rotated x, y, z.
    """
    return [vec[0]*np.cos(theta*_DEG2RA) - vec[2]*np.sin(theta*_DEG2RA),
            vec[1][None,:],
            vec[2]*np.cos(theta*_DEG2RA) + vec[0]*np.sin(theta*_DEG2RA)]

def rot_xz_sph(l, b, theta): # noqa: E741
    """Rotate spherical coordinates (l, b) by angle theta around axis (0, 1, 0).

    Calls :func:`sph2cart`, :func:`rot_xz` and :func:`cart2sph`.

    Parameters
    ----------
    l, b : float or array_like
        Spherical coordinates (theta, phi), in degrees.

    theta : float or array_like
        Rotation angle, in degrees.

    Returns
    -------
    numpy.ndarray
        Rotated [theta, phi], in degrees.
    """
    v_rot = rot_xz( sph2cart([1,l,b]), theta)
    return cart2sph(v_rot)[1:]
