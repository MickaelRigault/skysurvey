"""HEALPix-based functions and classes for handling survey observations."""

from .core import BaseSurvey

import pandas
import numpy as np
import healpy as hp
import warnings


def get_ipix_in_range(nside, ra_range=None, dec_range=None, in_rad=False):
    """Get the healpix pixel indices (ipix) within the given ra and dec range.

    Parameters
    ----------
    nside : int
        Healpix nside.

    ra_range, dec_range : array_like or None, optional
        Min and max to define a coordinate range to be considered.
        If None, no limit. The default is None.

    in_rad : bool, optional
        Whether the ra and dec ranges are in radian (True) or degree (False).
        The default is False.

    Returns
    -------
    numpy.ndarray
        Healpix pixel indices.
    """
    npix = hp.nside2npix(nside)
    pixs = np.arange(npix) # list of all healpix pixels
    if ra_range is None and dec_range is None:
        return pixs
    
    ras,decs = hp.pix2ang(nside, pixs)
    ras = (np.pi/2-ras)
    # only dec range
    if ra_range is None:
        if not in_rad:
            dec_range = np.multiply(dec_range, np.pi/180) # works if list given
        return pixs[(decs>=dec_range[0]) & (decs<=dec_range[1])]
    
    # only ra range    
    if dec_range is None:
        if not in_rad:
            ra_range = np.multiply(ra_range, np.pi/180) # works if list given
        return pixs[(ras>=ra_range[0]) & (ras<=ra_range[1])]
    
    # both
    if not in_rad:
        ra_range = np.multiply(ra_range, np.pi/180) # works if list given
        dec_range = np.multiply(dec_range, np.pi/180) # works if list given
    return pixs[(ras>=ra_range[0]) & (ras<=ra_range[1]) & (decs>=dec_range[0]) & (decs<=dec_range[1])]


# ================== #
#                    #
#    Healpix         #
#                    #
# ================== #
class HealpixSurvey( BaseSurvey ):
    """Survey whose fields are healpix pixels.

    Parameters
    ----------
    nside : int
        Healpix nside parameter.

    data : pandas.DataFrame or None, optional
        Observing data. The default is None.

    See Also
    --------
    from_data : Load the instance given observing data.
    from_random : Generate random observing data and load the instance.
    """
   
    def __init__(self, nside, data=None):
        """Initialize the HealpixSurvey class."""
        super().__init__(data)
        self._nside = nside
        
    @classmethod
    def from_data(cls, nside, data):
        """Load an instance given survey data and healpix size (nside).

        Parameters
        ----------
        nside : int
            Healpix nside parameter.

        data : pandas.DataFrame
            Observing data.

        Returns
        -------
        HealpixSurvey
            The loaded instance.

        See Also
        --------
        from_random : Generate random observing data and load the instance.
        """
        return cls(nside=nside, data=data)

    @classmethod
    def from_random(cls, nside, size, 
                    bands, 
                    mjd_range, skynoise_range,
                    ra_range=None, dec_range=None,
                    rng=None, **kwargs):
        """Load an instance with random observing data.

        Parameters
        ----------
        nside : int
            Healpix nside parameter.

        size : int
            Number of observations to draw.

        bands : list of str
            List of bands that should be drawn.

        mjd_range : array_like
            Min and max mjd for the random drawing.

        skynoise_range : array_like
            Min and max skynoise for the random drawing.

        ra_range, dec_range : array_like or None, optional
            Min and max to define a coordinate range to be considered.
            If None, no limit. The default is None.

        rng : None, int, or numpy.random.Generator, optional
            Seed for the random number generator (see
            :func:`numpy.random.default_rng`). If None, an unpredictable entropy
            is pulled from the OS. If an int (>0), it sets the initial
            `BitGenerator` state. If a Generator, it is used unaltered.
            The default is None.

        **kwargs
            Passed to :meth:`draw_random`.

        Returns
        -------
        HealpixSurvey
            The loaded instance.
        """
        this = cls(nside=nside)
        this.draw_random(size,  bands,  
                        mjd_range, skynoise_range, 
                        ra_range=ra_range, dec_range=dec_range,
                        inplace=True, rng=rng, **kwargs)
        return this

    @classmethod
    def from_pointings(cls, nside, data,
                       footprint=None, moc=None,
                       rakey="ra", deckey="dec",
                       backend="polars",
                       use_pyarrow_extension_array=False,
                       **kwargs):
        """Load an instance given observing pointings of a survey.

        This loads a :class:`~skysurvey.survey.polygon.PolygonSurvey` using its
        ``from_pointings`` method and converts it into a healpix survey using its
        ``to_healpix()`` method.

        Parameters
        ----------
        nside : int
            Healpix nside parameter.

        data : pandas.DataFrame or dict
            Observing data, must contain the `rakey` and `deckey` columns.

        footprint : shapely.geometry.Polygon or None, optional
            Footprint in the sky of the observing camera. The default is None.

        moc : mocpy.MOC or None, optional
            MOC representation of the observing camera (used if `footprint` is
            None). The default is None.

        rakey : str, optional
            Name of the R.A. column (in deg). The default is 'ra'.

        deckey : str, optional
            Name of the declination column (in deg). The default is 'dec'.

        backend : {'polars', 'pandas', 'dask'}, optional
            Which backend to use to merge the data (speed issue):

            - 'polars' (fastest): requires polars installed; converted to pandas
              at the end.
            - 'pandas' (classic): the normal way.
            - 'dask' (lazy): a persisted dask.dataframe is returned.

            The default is 'polars'.

        use_pyarrow_extension_array : bool, optional
            Ignored if backend is not 'polars'. Should the pandas dataframe be
            based on pyarrow arrays (like in polars; faster to load, but
            numpy.asarray will be used by pandas when needed, which will then slow
            things down) rather than numpy arrays (slow to load but faster then).
            The default is False.

        **kwargs
            Passed to
            :meth:`~skysurvey.survey.polygon.PolygonSurvey.from_pointings`.

        Returns
        -------
        HealpixSurvey
            The loaded instance.
        """
        from .polygon import PolygonSurvey
        # Create a generic polygon survey
        polysurvey = PolygonSurvey.from_pointings(data, footprint=footprint,
                                                  moc=moc,
                                                  rakey=rakey, deckey=deckey,
                                                  **kwargs)
        # convert it to healpix
        return polysurvey.to_healpix(nside, backend=backend,
                                         pass_data=True,
                                         polars_to_pandas=True,
                                         use_pyarrow_extension_array=use_pyarrow_extension_array)
        
    
    # ============== #
    #   Methods      #
    # ============== #
    def get_field_area(self):
        """Get the area (in deg**2) of a healpix pixel.

        Returns
        -------
        float
            Pixel area in deg**2.
        """
        return hp.nside2pixarea(self.nside, degrees = True)
    
    def get_observed_area(self, min_obs=1):
        """Get the observed area (in deg**2).

        A healpix pixel is considered observed if present more than `min_obs`
        times (at least once if `min_obs` <= 1).

        Parameters
        ----------
        min_obs : int, optional
            Minimum number of observations to consider a field as observed.
            The default is 1.

        Returns
        -------
        float
            Observed area in deg**2.
        """
        if min_obs <=1: # 0 or 1 the same
            nfields = self.data["fieldid"].nunique()
        else:
            nobs = self.data["fieldid"].value_counts()
            nfields = len(nobs[nobs>min_obs])
        
        return self.get_field_area() * nfields

    def get_polygons(self, observed_fields=False, as_vertices=False, origin=180):
        """Get the list of field polygons.

        Parameters
        ----------
        observed_fields : bool, optional
            Should this be limited to observed fields? The default is False.

        as_vertices : bool, optional
            Should this return a list of shapely.geometry.Polygon (False) or
            their vertices (True; shape N [fields], 2 [ra, dec], 4 [corners]).
            The default is False.

        origin : float, optional
            Origin of the R.A. coordinate (center of image). The default is 180.

        Returns
        -------
        list of shapely.geometry.Polygon or numpy.ndarray
            The polygons, or their vertices if `as_vertices` is True.
        """
        if observed_fields:
            fieldid = self.data[self.fieldids.name].unique()
        else:
            fieldid = self.fieldids

        corners = hp.boundaries(nside=self.nside, pix=fieldid)
        corners = np.moveaxis(corners,1,2)
        ang = np.asarray([hp.vec2ang(corners_, lonlat=True) for corners_ in corners])
        ang[:,0,:] = (origin-ang[:,0,:])%360 # set back origin

        if as_vertices:
            return ang
        
        from shapely import geometry
        polygons = [geometry.Polygon(ang_.T) for ang_ in ang]
        return polygons

    def get_skyarea(self, as_multipolygon=True, buffer=0.01):
        """Get the multipolygon (or union) of the observed field geometries.

        Parameters
        ----------
        as_multipolygon : bool, optional
            If True, returns a multipolygon. Otherwise, returns the unary_union
            of the polygons. The default is True.

        buffer : float or None, optional
            Buffer (in deg) around the polygons. This helps joining edges and
            reduces the number of isolated sky-pixels which may artificially slow
            down computation. If None, no buffer is applied. The default is 0.01.

        Returns
        -------
        shapely.geometry.MultiPolygon or shapely.geometry.Polygon
            The sky area.
        """
        from shapely import ops, geometry
        
        ps = self.get_polygons(observed_fields=True, as_vertices=False)
        if as_multipolygon:
            skyarea = geometry.MultiPolygon(ps)
        else:
            skyarea = ops.unary_union(ps)
            
        if buffer is not None:
            skyarea = skyarea.buffer(buffer)
            
        return skyarea
    # ------- #
    #  core   #
    # ------- #

    def radec_to_fieldid(self, radec, origin=180, observed_fields=False):
        """Get the fieldid associated to the given coordinates.

        Parameters
        ----------
        radec : pandas.DataFrame or array_like
            Coordinates in degree, either a DataFrame with 'ra' and 'dec' columns
            or a (ra, dec) array.

        origin : float, optional
            Value of the central R.A. The default is 180.

        observed_fields : bool, optional
            Should this be limited to fields actually observed?
            The default is False.

        Returns
        -------
        pandas.DataFrame
            Fieldid for each input coordinate (indexed as the input).
        """
        if type(radec) is pandas.DataFrame:
            ra = np.asarray(radec["ra"].values, dtype="float")
            dec = np.asarray(radec["dec"].values, dtype="float")
            index = radec.index.copy()
            if index.name is None:
                index.name = "index_radec"
            
        else:
            ra, dec = np.atleast_1d(radec)
            ra = np.atleast_1d(ra)
            dec = np.atleast_1d(dec)
            index = pandas.Index(np.arange( len(ra) ), name="index_radec")
            
        fields = hp.ang2pix(self.nside, (90 - dec) * np.pi/180, (origin-ra) * np.pi/180)
        df = pandas.DataFrame(fields, columns = [self.fieldids.name], index=index)
        
        if observed_fields:
            observed_fields = self.data[self.fieldids.name].unique()
            df = df[df[self.fieldids.name].isin(observed_fields)]
        
        return df

    def get_field_centroid(self, origin=180):
        """Get the centroid of the fields.

        Parameters
        ----------
        origin : float, optional
            Origin of the ra coordinates. The default is 180.

        Returns
        -------
        ra : numpy.ndarray
            R.A. of the field centroids (in deg).

        dec : numpy.ndarray
            Declination of the field centroids (in deg).
        """
        dec, ra = np.asarray(hp.pix2ang(self.nside, self.fieldids))*180/np.pi
        dec = 90-dec
        ra = (origin-ra)%360
        return ra, dec

    # ------- #
    #  draw   #
    # ------- #    
    def draw_random(self, size, 
                    bands, mjd_range, skynoise_range,
                    gain_range=1, zp_range=25,
                    ra_range=None, dec_range=None,
                    inplace=False, nside=None,
                    rng=None, **kwargs):
        """Draw random observations.

        Parameters
        ----------
        size : int
            Number of observations to draw.

        bands : list of str
            List of bands that should be drawn.

        mjd_range : array_like
            Min and max mjd for the random drawing.

        skynoise_range, gain_range, zp_range : array_like, float or int
            Range to be considered. If float or int, this value will always be
            used. Otherwise, a uniform distribution within the range is assumed.
            The defaults are 1 for `gain_range` and 25 for `zp_range`.

        ra_range, dec_range : array_like or None, optional
            Min and max to define a coordinate range to be considered.
            If None, no limit. The default is None.

        inplace : bool, optional
            If True, replace the current `self.data`. Otherwise, return a new
            instance of the class with the generated observing data.
            The default is False.

        nside : int or None, optional
            New healpix nside parameter. If None, the current nside is used.
            If given with `inplace=True`, a warning is raised and a new instance
            is returned (inplace is set to False). The default is None.

        rng : None, int, or numpy.random.Generator, optional
            Seed for the random number generator (see
            :func:`numpy.random.default_rng`). If None, an unpredictable entropy
            is pulled from the OS. If an int (>0), it sets the initial
            `BitGenerator` state. If a Generator, it is used unaltered.
            The default is None.

        **kwargs
            Passed to :meth:`_draw_random`.

        Returns
        -------
        HealpixSurvey or None
            New instance if `inplace` is False, None otherwise.

        See Also
        --------
        from_random : Generate random observing data and load the instance.
        set_data : Set the observing data to the instance.
        """
        if nside is None: # don't change nside
            nside = self.nside
            
        elif inplace: # change nside
            warnings.warn("Cannot change nside with inplace=True, a copy (inplace=False) is returned.")
            inplace = False
            
        data = self._draw_random(nside, size, 
                                 bands, mjd_range, skynoise_range, 
                                 ra_range=ra_range, dec_range=dec_range,
                                 gain_range=gain_range, zp_range=zp_range,
                                 rng=rng,
                                 **kwargs)
        
        if not inplace:
            return self.__class__.from_data(nside=nside, data=data)

        self.set_data(data)
        
    # ----------- #
    #  PLOTTER    #
    # ----------- #
    def show(self, stat='size', column=None, title=None, data=None, vmin=None,
             vmax=None, seed=None, **kwargs):
        """Show the sky coverage using `healpy.mollview`.

        Parameters
        ----------
        stat : str, optional
            Element passed to `groupby.agg()`, e.g. 'mean', 'std', etc.
            If stat is 'size', this returns the number of observations per field.
            The default is 'size'.

        column : str or None, optional
            Column of the dataframe the stat should be applied to.
            Ignored if stat is 'size'. The default is None.

        title : str or None, optional
            Title of the `healpy.mollview` plot. The default is None.

        data : pandas.Series, dict, array_like or None, optional
            Values to plot per field. If None, the field statistic is computed
            from `self.data` (or random values are drawn if there is no data).
            Leave to None if unsure. The default is None.

        vmin, vmax : float or None, optional
            Values below `vmin` (above `vmax`) are clipped. The default is None.

        seed : int or None, optional
            Random seed used to draw values if there is no data.
            The default is None.

        **kwargs
            Passed to :func:`healpy.mollview`.

        See Also
        --------
        get_fieldstat : Get observing statistics for the fields.
        """
        if data is None:
            if self.data is None:
                rng = np.random.default_rng(seed=seed) 
                data = rng.uniform(size=self.nfields)
            else:
                data = self.get_fieldstat(stat=stat, columns=column,
                                              incl_zeros=True, fillna=np.nan,
                                              data=data)
                
        else:
            if type(data) is dict:
                data = pandas.Series(data)
                
            if type(data) is pandas.Series:
                data = data.reindex(self.fieldids).values

        if vmin is not None:
            data[data<vmin] = vmin
            
        if vmax is not None:
             data[data>vmax] = vmax
             
        return hp.mollview(data, title=title, **kwargs)
        
    # ============== #
    # Static Methods #
    # ============== #        
    @staticmethod
    def _draw_random(nside, size, 
                     bands,  
                     mjd_range, skynoise_range,
                     gain_range=1,
                     zp_range=[27,30],
                     ra_range=None, dec_range=None,
                     rng=None):
        """Draw random observations (internal).

        Parameters
        ----------
        nside : int
            Healpix nside parameter.

        size : int
            Number of observations to draw.

        bands : list of str
            List of bands that should be drawn.

        mjd_range : array_like
            Min and max mjd for the random drawing.

        skynoise_range : array_like
            Min and max skynoise for the random drawing.

        gain_range : array_like or float, optional
            Min and max gain for the random drawing. The default is 1.

        zp_range : array_like or float, optional
            Min and max zp for the random drawing. The default is [27, 30].

        ra_range, dec_range : array_like or None, optional
            Min and max to define a coordinate range to be considered.
            If None, no limit. The default is None.

        rng : None, int, or numpy.random.Generator, optional
            Seed for the random number generator (see
            :func:`numpy.random.default_rng`). If None, an unpredictable entropy
            is pulled from the OS. If an int (>0), it sets the initial
            `BitGenerator` state. If a Generator, it is used unaltered.
            The default is None.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with the drawn observations.
        """
        rng = np.random.default_rng(rng)
        # np.resize(1, 2) -> [1,1]
        mjd = rng.uniform(*np.resize(mjd_range,2), size=size)
        band = rng.choice(bands, size=size)
        skynoise = rng.uniform(*np.resize(skynoise_range, 2), size=size)
        gain = rng.uniform(*np.resize(gain_range, 2), size=size)
        zp = rng.uniform(*np.resize(zp_range, 2), size=size)
        # = coords
        # no radec limit
        if ra_range is None and dec_range is None:
            npix = hp.nside2npix(nside)
            ipix = rng.uniform(0, npix, size=size)
        else:
            ipix_ok = get_ipix_in_range(nside, ra_range=ra_range, dec_range=dec_range)
            ipix = rng.choice(ipix_ok, size=size)
            
        # data sorted by mjd
        data = pandas.DataFrame(zip(mjd, band, skynoise, gain, zp, ipix),
                               columns=["mjd","band","skynoise", "gain", "zp","fieldid"]
                               ).sort_values("mjd"
                               ).reset_index(drop=False) # don't need to know the creation order
        return data
    
    # ============== #
    #   Properties   #
    # ============== #
    @property
    def nside(self):
        """Healpix nside parameter (defines the fields' size and number)."""
        return self._nside
    
    @property
    def nfields(self):
        """Number of fields (shortcut to npix)."""
        return self.npix
    
    @property    
    def npix(self):
        """Number of healpix pixels."""
        if not hasattr(self, "_npix") or self._npix is None:
            self._npix = hp.nside2npix(self.nside)
            
        return self._npix

    @property
    def fieldids(self):
        """Id of the individual fields."""
        fieldids = np.arange( self.npix )
        # use pandas.index for self consistency with polygon.survey
        return pandas.Index(fieldids, name="fieldid")
   
    def metadata(self):
        """Get the metadata information.

        Returns
        -------
        dict
            Survey metadata, including the nside.
        """
        meta = super().metadata
        meta["nside"] = self.nside
        return meta
