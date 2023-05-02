import warnings
import numpy as np
import pandas

__all__ = ["BaseSurvey"] # no import when 'import *'

class BaseSurvey( object ):
    
    REQUIRED_COLUMNS = ['mjd', 'band', 'skynoise', 'fieldid', "gain", "zp"]
    
    def __init__(self, data):
        """ 
        See also
        --------
        
        """
        self.set_data(data)
    
    def __array__(self):
        """ """
        return self.data.__array__()
    
    # ============== #
    #   Methods      #
    # ============== #    
    def set_data(self, data, lower_precision=True):
        """ set the observing data 

        = It is unlikely you need to use that directly. =

        Parameters
        ----------
        data: pandas.DataFrame
             observing data. see REQUIRED_COLUMNS for the list of
             required columns.

        lower_precision: bool
             change the types from 64 to 32 precision when possible.

        Returns
        -------
        None
        """
        if data is not None and not np.in1d(self.REQUIRED_COLUMNS, data.columns).all():
            raise ValueError(f"at least one of the following column name if missing {self.REQUIRED_COLUMNS}")

        if lower_precision and data is not None:
            data = data.astype( {k: str(v).replace("64","32") for k, v in data.dtypes.to_dict().items()})
            
        self._data = data
        
    # ------------ #
    #   GETTER     #
    # ------------ #
    def get_fieldcoverage(self, incl_zeros=False, fillna=np.NaN,
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
                        incl_zeros=False, fillna=np.NaN,
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
            
        if stat in ["size","value_counts"]:
            data = data["fieldid"].value_counts()
            
        elif columns is None:
            data = data.groupby("fieldid").agg(stat)
        else:
            data = data.groupby("fieldid")[columns].agg(stat)
            
        if not incl_zeros:
            return data
        
        if type(data) == pandas.Series: # Serie
            all_data = np.ones(self.nfields)*fillna
            all_data[data.index] = data.values
            all_data = pandas.Series(all_data, name=f"fieldid_{stat}")
            
        else: # DataFrame
            all_data = np.ones((self.nfields, data.shape[1]))*fillna
            all_data[data.index] = data.values
            all_data = pandas.DataFrame(all_data, columns=data.columns)
            
        return all_data
        
        
    def radec_to_fieldid(self, radec):
        """ get the fieldid of the given (list of) coordinates """
        raise NotImplementedError("you have not implemented radec_to_fieldid for your survey")
    
    # ----------- #
    #  PLOTTER    #
    # ----------- #        
    def show(self):
        """ shows the sky coverage """
        raise NotImplementedError("you have not implemented show for your survey")


    def show_nexposures(self, ax=None, exposure_key="expid",
                            bands=None,perband=True, band_key="band", band_colors=None,
                            fieldid=None,
                            legend=True, **kwargs):
        """ """
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
        print(nobs.max())
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
    def of_type(self):
        """ kind of survey that is """
        return str(type(self)).split("'")[-2].split(".")[-1]

    @property
    def date_range(self):
        """ first and last date of the survey """
        return np.min(self.data["mjd"]), np.max(self.data["mjd"])
        
