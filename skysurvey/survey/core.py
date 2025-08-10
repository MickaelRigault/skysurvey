import warnings
import numpy as np
import pandas

__all__ = ["BaseSurvey"] # no import when 'import *'

class BaseSurvey( object ):
    
    REQUIRED_COLUMNS = ['mjd', 'band', 'skynoise', "gain", "zp"]

    # NOTE 
    # -----
    # ``skynoise`` is the image background contribution to the flux measurement
    # error (in units corresponding to the specified zeropoint and zeropoint
    # system). To get the error on a given measurement, ``skynoise`` is added
    # in quadrature to the photon noise from the source.
    #
    # It is left up to the user to calculate ``skynoise`` as they see fit as the
    # details depend on how photometry is done and possibly how the PSF is
    # is modeled. As a simple example, assuming a Gaussian PSF, and perfect
    # PSF photometry, ``skynoise`` would be ``4 * pi * sigma_PSF * sigma_pixel``
    # where ``sigma_PSF`` is the standard deviation of the PSF in pixels and
    # ``sigma_pixel`` is the background noise in a single pixel in counts.
    # -- note from sncosmo

    
    
    def __init__(self, data):
        """ 
        Initialize the BaseSurvey class.

        Parameters
        ----------
        data: pandas.DataFrame
            observing data.
        """
        self.set_data(data)
    
    def __array__(self):
        """ numpy array representation of the data """
        return self.data.__array__()
    
    # ============== #
    #   Methods      #
    # ============== #    
    def set_data(self, data, lower_precision=True, sort_mjd=True):
        """ set the observing data 

        = It is unlikely you need to use that directly. =

        Parameters
        ----------
        data: pandas.DataFrame
            observing data. see REQUIRED_COLUMNS for the list of
            required columns.

        lower_precision: bool
            change the types from 64 to 32 precision when possible.

        sort_mjd: bool
            should this sort by mjd (if needed) as required to draw dataset
            
        Returns
        -------
        None
        """
        if data is None:
            self._data = None
            return
        
        if not np.in1d(self.REQUIRED_COLUMNS, data.columns).all():
            warnings.warn(f"at least one of the following column name if missing {self.REQUIRED_COLUMNS}")

        if self.fields is not None and self.fieldids.name is not None:
            if not np.all([f_name in data for f_name in self.fieldids.names]):
                warnings.warn(f"fieldid {self.fieldids.names} are not in the input data")
            
        if lower_precision:
            data = data.astype( {k: str(v).replace("64","32") for k, v in data.dtypes.to_dict().items()})

        if sort_mjd and not (data["mjd"].is_monotonic_increasing or data["mjd"].is_monotonic_decreasing):
            data = data.sort_values("mjd")
            
        self._data = data
    # ------------ #
    #   GETTER     #
    # ------------ #
    def get_timerange(self, timekey="mjd"):
        """ returns the min and max of the given timekey column.

        Parameters
        ----------
        timekey: str
            column name of the time column.

        Returns
        -------
        numpy.array
        """
        return self.data[timekey].agg(["min", "max"]).values
        
    def get_fieldcoverage(self, incl_zeros=False, fillna=np.nan,
                          **kwargs):
        """ short cut to get_fieldstat('size') 

        Parameters
        ----------
        incl_zeros: bool
            fields will no entries will not be shown 
            except if incl_zeros is True

        fillna: float, str
            format of the N/A entries

        **kwargs goes to get_fieldstat()

        Returns
        -------
        DataFrame or Serie 
            following groupby.agg()

        See also
        --------
        get_fieldstat: get observing statistics for the fields

        """
        return self.get_fieldstat(stat="size", columns=None,
                                    incl_zeros=incl_zeros, 
                                  fillna=fillna, **kwargs)
    
    def get_fieldstat(self, stat, columns=None,
                        incl_zeros=False, fillna=np.nan,
                        data=None):
        """ get observing statistics for the fields

        basically a shortcut to ``data.groupby("fieldid")[`column`].`stat`()`` 
        
        Parameters
        ----------
        stat: str, list
            element to be passed to groupby.agg() 
            could be e.g.: 'mean' or ['mean', 'std'] or [np.median, 'mean'] etc.
            If stat = 'size', this returns data["fieldid"].value_counts()
            (slightly faster than groupby("fieldid").size()).
                
        columns: str, list, None
            name of the columns to be kept.
            None means no cut.

        incl_zeros: bool
            fields will no entries will not be shown 
            except if incl_zeros is True

        fillna: float, str
            format of the N/A entries
            
        data: pandas.DataFrame, None
            data you want this to be applied to.
            if None, a copy of self.data is used.
            = leave to None if unsure =
        
        Returns
        -------
        DataFrame or Serie 
            following groupby.agg()

        """
        if data is None:
            data = self.data.copy()

        fieldids = self.fieldids.names

        fieldgrouped = self.data.groupby(fieldids)
        if stat in ["size","value_counts"]:
            data = fieldgrouped.size()
            
        elif columns is None:
            data = fieldgrouped.agg(stat)
        else:
            data = fieldgrouped[columns].agg(stat)
            
        if not incl_zeros:
            return data

        return data.reindex(self.fieldids, level=0)
        
        
    def radec_to_fieldid(self, radec):
        """ get the fieldid of the given (list of) coordinates

        Parameters
        ----------
        radec: pandas.DataFrame or 2d array
            coordinates in degree

        Returns
        -------
        pandas.Series
        """
        raise NotImplementedError("you have not implemented radec_to_fieldid for your survey")

    def get_observations_from_coords(self, radec):
        """ returns the data associated to the input radec coordinates
        
        (calls radec_to_fieldid and select data matching the fieldid)

        Parameters
        ----------
        radec: pandas.DataFrame or 2d array
            coordinates in degree
            (see format radec_to_fieldid())
            
        Returns
        -------
        pandas.DataFrame
            copy of the data observed in the given radec coordinates
        
        """
        fields = self.radec_to_fieldid(radec, observed_fields=True)
        return self.data[ self.data[self.fieldids.name].isin(fields[self.fieldids.name]) ].copy()
    
    # ----------- #
    #  PLOTTER    #
    # ----------- #        
    def show(self):
        """ shows the sky coverage.

        Raises
        ------
        NotImplementedError
            This method is not implemented for this survey.
        """
        raise NotImplementedError("you have not implemented show for your survey")


    def show_nexposures(self, ax=None, exposure_key="expid",
                            bands=None,perband=True, band_key="band", band_colors=None,
                            fieldid=None,
                            legend=True, **kwargs):
        """ show the number of exposures per day.

        Parameters
        ----------
        ax: matplotlib.axes
            axes to plot on.

        exposure_key: str
            column name of the exposure id.

        bands: list
            list of bands to plot.

        perband: bool
            if True, plot the number of exposures per band.

        band_key: str
            column name of the band.

        band_colors: dict
            dictionary of colors for each band.

        fieldid: int or list
            field id to plot.

        legend: bool
            if True, show the legend.

        **kwargs goes to ax.bar

        Returns
        -------
        matplotlib.figure
        """
        from astropy.time import Time
        
        day = self.data["mjd"].astype(int)
        day.name = "day"
        data = self.data.join(day) # this is a copy

        if fieldid is not None:
            data = data[data["fieldid"].isin( np.atleast_1d(fieldid) )]
        
        if ax is None:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(9,2))
            ax = fig.add_axes([0.075, 0.2, 0.85, 0.6])
        else:
            fig = ax.figure

        if not perband:
            nobs = data.groupby(exposure_key).first().groupby("day").size()
            max_obs = nobs.max()
            all_days = nobs.index
            nbands = 1
        else:
            nobs = data.groupby(exposure_key).first().groupby(["day", band_key]).size()
            max_obs = nobs.groupby(level=0).sum().max()
            all_days = nobs.index.levels[0]
            if bands is None:
                bands = nobs.index.levels[1]
            nbands = len(bands)

        times = Time(all_days.astype( float ), format="mjd").datetime

        # plotting properties
        prop = {**dict(zorder=3, width=0.95), **kwargs}

        if perband:
            bottom = 0
            if band_colors is None:
                band_colors = [None for i_ in range(len(bands))]
                
            for band_, color_ in zip( bands, band_colors ):
                d_ = nobs.xs(band_, level=1).reindex(all_days).fillna(0).astype(int).values
                ax.bar(times, d_, color=color_,
                       bottom=bottom, label=f"{band_}",
                       **prop)
                bottom += d_
        else:
             ax.bar(times, nobs.values, **prop)

        from matplotlib import dates as mdates
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)


        clearwhich = ["left","right","top"] # "bottom"
        [ax.spines[which].set_visible(False) for which in clearwhich]
        ax.tick_params(axis="y", labelsize="small", 
                                   labelcolor="0.7", color="0.7")

        ax.grid(axis="y", lw=0.5, color='0.7', zorder=1, alpha=0.5)
        ax.set_ylabel("exposures per day", color="0.7", fontsize="small")

        ax.set_ylim(ymin=0, ymax=np.round(max_obs*1.05,decimals=-1) )
        if legend:
            ax.legend(loc=[0,1], ncol=nbands, frameon=False, fontsize="small")
            
        return fig
    
    # ============== #
    #   Properties   #
    # ============== #
    @property
    def data(self):
        """ dataframe containing what has been observed when
        aka. the observing data 
        """
        return self._data
    
    @property
    def metadata(self):
        """ metadata associated to the survey """
        meta = {"type":self.of_type}
        return meta
    
    @property    
    def nfields(self):
        """ number of fields """
        if not hasattr(self,"_nfields") or self._nfields is None:
            warnings.warn("no nfields set, so this is assuming max of data['fieldid'].")
            self._nfields =self.data["fieldid"].max()
            
        return self._nfields

    @property
    def fields(self):
        """ geodataframe containing the fields coordinates """
        if not hasattr(self,"_fields"):
            return None
        return self._fields

    
    @property
    def of_type(self):
        """ kind of survey that is """
        return str(type(self)).split("'")[-2].split(".")[-1]

    @property
    def date_range(self):
        """ first and last date of the survey """
        return np.min(self.data["mjd"]), np.max(self.data["mjd"])
        
