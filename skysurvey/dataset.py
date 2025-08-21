""" This library concerns the data as observed """

#
import pandas
import numpy as np
from copy import copy

#
import sncosmo
from astropy.table import Table

from .template import Template
from .lightcurves import get_obsdata, _get_obsdata_

__all__ = ["DataSet"]

# ================== #
#                    #
#    DataSet         #
#                    #
# ================== #

class DataSet(object):
    """ A class for managing and realistic transient light curves given true data and survey observing logs.

    This class provides methods to load, manipulate, and visualize light curve data
    based on target and survey information.

    Parameters
    ----------
    data : pandas.DataFrame
        Multi-index dataframe corresponding to the concatenation of all targets observations.

    targets : skysurvey.Target or child of, optional
        Target data corresponding to the true target parameters (as given by nature).

    survey : skysurvey.Survey or child of, optional
        Survey that has been used to generate the dataset (if known).

    Attributes
    ----------
    data : pandas.DataFrame
        Light curve data as observed by the survey.

    targets : skysurvey.Target or child of
        Target data corresponding to the true target parameters.

    survey : skysurvey.Survey or child of
        Survey used to generate the dataset.

    obs_index : pandas.Index
        Index of the observed targets.

    See Also
    --------
    from_targets_and_survey : Loads a dataset (observed data) given targets and survey.
    read_parquet : Loads a stored dataset.

    Methods
    -------
    from_targets_and_survey(targets, survey, client=None, incl_error=True, **kwargs)
        Loads a dataset given targets and a survey.

    read_parquet(parquetfile, survey=None, targets=None, **kwargs)
        Loads a stored dataset from a parquet file.

    read_from_directory(dirname, **kwargs)
        Loads a directory containing the dataset, the survey, and the targets.

    set_data(data)
        Sets the light curve data.

    set_targets(targets)
        Sets the targets.

    set_survey(survey)
        Sets the survey.

    get_data(add_phase=False, phase_range=None, index=None, redshift_key="z", detection=None, zp=None)
        Accesses the data with additional tools.

    get_ndetection(phase_range=None, per_band=False)
        Gets the number of detections for each light curve.

    get_target_lightcurve(index, detection=None, phase_range=None)
        Gets the observation of the given target.

    show_target_lightcurve(ax=None, fig=None, index=None, zp=25, lc_prop={}, bands=None, show_truth=True, format_time=True, t0_format="mjd", phase_window=None, **kwargs)
        Plots the light curve of a target.

    realize_survey_target_lcs(targets, survey, template_prop={}, nfirst=None, incl_error=True, client=None, discard_bands=False, trim_observations=False, phase_range=None)
        Creates the light curve of the input targets as they would be observed by the survey.

    _realize_survey_kindtarget_lcs(targets, survey, template_prop={}, nfirst=None, incl_error=True, client=None, discard_bands=False, trim_observations=False, phase_range=None)
        Creates the light curve of the input single-kind targets as they would be observed by the survey.
    """
    
    def __init__(self, data, targets=None, survey=None):
        """ Initialize the DataSet class.
        
        The classmethod Dataset.from_targets_and_survey() should be favored 
        for loading the dataset.
        
        Parameters
        ----------
        data : pandas.DataFrame
            Multi-index dataframe corresponding to the concatenation of all targets observations.
        targets : skysurvey.Target or child of, optional
            Target data corresponding to the true target parameters (as given by nature).
        survey : skysurvey.Survey or child of, optional
            Survey that has been used to generate the dataset (if known).
       
        See also
        --------
        from_targets_and_survey: loads a dataset (observed data) given targets and survey
        read_parquet: loads a stored dataset
        """
        self.set_data(data)
        self.set_targets(targets)
        self.set_survey(survey)
        
    @classmethod
    def from_targets_and_survey(cls, targets, survey, client=None,
                                    incl_error=True, **kwargs):
        """ loads a dataset (observed data) given targets and a survey

        This first matches the targets (given targets.data[["ra","dec"]]) with the
        survey to find which target has been observed with which field.
        Then simulate the targets lightcurves given the observing data (survey.data).
        

        Parameters
        ----------
        targets: skysurvey.Target (or child of)
            target data corresponding to the true target parameters  
            (as given by nature)
            
        survey: skysurvey.Survey, skysurvey.GridSurvey (or child of)
            sky observation (what was observed when with which situation)

        client: dask.distributed.Client()
            dask client to use if any. This is used for 
            parallelization of the lc generation per field.

        incl_error: bool

        **kwargs goes to realize_survey_target_lcs

        Returns
        -------
        class instance
            the observation data have been derived and stored as self.data

        See also
        --------
        read_parquet: loads a stored dataset
        """
        from .tools.speedutils import eff_concat
        lightcurves, fieldids = cls.realize_survey_target_lcs(targets, survey,
                                                                  client=client,
                                                                  incl_error=incl_error,
                                                                  **kwargs)
        chunk_size = int(np.sqrt( len(lightcurves) )) # good guess
        return cls( eff_concat(lightcurves, chunk_size=chunk_size, keys=fieldids).reset_index(survey.fieldids.names),
                       targets=targets, survey=survey)

    @classmethod
    def read_parquet(cls, parquetfile, survey=None, targets=None, **kwargs):
        """ loads a stored dataset. 

        Only the observation data can be loaded this way, 
        not the survey nor the targets (truth). 

        Parameters
        ----------
        parquetfile: str
            path to the parquet file containing the dataset (pandas.DataFrame)

        survey: skysurvey.Survey (or child of), None
            survey that have been used to generate the dataset (if you know it)

        targets: skysurvey.Target (of child of), None
            target data corresponding to the true target parameters 
            (as given by nature)

        **kwargs goes to pandas.read_parquet

        Returns
        -------
        class instance
            with a dataset loaded but maybe no survey nor targets

        See also
        --------
        from_targets_and_survey: loads a dataset (observed data) given targets and survey
        """
        data = pandas.read_parquet(parquetfile, **kwargs)
        return cls(data, survey=survey, targets=targets)

    @classmethod
    def read_from_directory(cls, dirname, **kwargs):
        """ loads a directory containing the dataset, the survey and the targets

        = Not Implemented Yet = 

        Parameters
        ----------
        dirname: str
            path to the directory.
            
        Returns
        -------
        class instance
            
        See also
        --------
        from_targets_and_survey: loads a dataset (observed data) given targets and survey
        read_parquet: loads a stored dataset
        """
        raise NotImplementedError("read_from_directory is not yet available.")
        
    # ============== #
    #   Method       #
    # ============== #
    # -------- #
    #  SETTER  #
    # -------- #
    def set_data(self, data):
        """ lightcurve data as observed by the survey

        = It is unlikely you need to use that directly. =

        Parameters
        ----------
        data: pandas.DataFrame
            multi-index dataframe ((id, observation index))
            corresponding the concat of all targets observations

        Returns
        -------
        None

        See also
        --------
        read_parquet: loads a stored dataset
        """
        self._data = data
        self._obs_index = None
        
    def set_targets(self, targets):
        """ set the targets

        = It is unlikely you need to use that directly. =

        Parameters
        ----------
        targets: skysurvey.Target (of child of), None
            target data corresponding to the true target parameters 
            (as given by nature)

        Returns
        -------
        None

        See also
        --------
        from_targets_and_survey: loads a dataset (observed data) given targets and survey
        """
        self._targets = targets

    def set_survey(self, survey):
        """ set the survey 

        = It is unlikely you need to use that directly. =

        Parameters
        ----------
        survey: skysurvey.Survey (or child of), None
            survey that have been used to generate the dataset (if you know it)

        Returns
        -------
        None

        See also
        --------
        from_targets_and_survey: loads a dataset (observed data) given targets and survey
        """
        self._survey = survey

    # -------- #
    #  GETTER  #
    # -------- #
    def get_data(self, add_phase=False, phase_range=None, index=None, redshift_key="z",
                detection=None, zp=None):
        """ tools to access the data with additional tools 

        Parameters
        ----------
        add_phase: bool
            should the phase information 'phase_obs' (obs-frame), 'phase' (rest-frame)
            be added to the dataframe assuming the input target's t0 and redshift ?

        phase_range: array
            min and max phases to be returned. Applied on phase (rest-frame).
            Setting this sets add_phase to True.

        index: pandas.Index, list, None
            select the index (targets id) you want.

        redshift_key: string
            name of the redshift column in the dset.targets.data. 
             = ignored if add_phase is False =

        detection: bool, None
            should this be limited to (non)detected points only ? 
            This follow the bool/None format:
            - detection=None: no selection
            - detection=False: only non-detected points
            - detection=True: onlyu detected points

        zp: float
            get the simulated data in the given zp system

        Returns
        -------
        pandas.DataFrame
        """
        if phase_range is not None:
            add_phase = True

        if index is not None:
            data = self.data.loc[index].copy()
        else:
            data = self.data.copy()
            index = data.index.levels[0]

        if add_phase:
            target_info = self.targets.data.loc[index][['t0', redshift_key]]
    #        target_info.index = self._data_index # for merging
            data["phase_obs"] = data["mjd"] - target_info["t0"]
            data["phase"] = data["phase_obs"]/(1+target_info[redshift_key])

        if phase_range is not None:
            data = data[data["phase"].between(*phase_range)]

        if detection is not None:
            flag_detection = (data["flux"]/data["fluxerr"])>=5
            if detection:
                data = data[flag_detection]
            else:
                data = data[~flag_detection]

        if zp is not None:
            coef = 10 ** (-(data["zp"].values - zp) / 2.5)
            data["flux"] *= coef
            data["fluxerr"] *= coef
            data["zp"] = zp
            
        return data
        
    def get_ndetection(self, phase_range=None, per_band=False):
        """ get the number of detection for each lightcurves

        Basically computes the number of datapoints with (flux/fluxerr)>detlimit)

        Parameters
        ----------
        phase_range: array
            rest-frame phase range to be considered.

        per_band: bool
            should be computation be made per band ?
            if true it will then be per target *and* per band.

        Returns
        -------
        pandas.Series
            the number of detected point per target (and per band if per_band=True)
        """
        
        data = self.get_data(phase_range=phase_range, detection=True)
        if per_band:
            groupby = [self._data_index, "band"]
        else:
            groupby = self._data_index
        
        ndetection = data.groupby(groupby).size()
        return ndetection

    def get_target_lightcurve(self, index, detection=None, phase_range=None):
        """ get the observation of the given target.
        
        = short cut to self.get_data(index=index) = 

        Parameters
        ----------
        detection: bool, None
            should this be limited to (non)detected points only ? 
            This follow the bool/None format:
            - detection=None: no selection
            - detection=False: only non-detected points
            - detection=True: onlyu detected points

        phase_range: array
            min and max phases to be returned. Applied on phase (rest-frame).
            Setting this sets add_phase to True.

        Returns
        -------
        pandas.DataFrame
            the lightcurve
        """
        return self.get_data(index=index,
                             phase_range=phase_range,
                             detection=detection)
        
    # -------- #
    #  PLOTTER #
    # -------- #
    def show_target_lightcurve(self, ax=None, fig=None, index=None, zp=25,
                                lc_prop={}, bands=None, show_truth=True,
                                format_time=True, t0_format="mjd", 
                                phase_window=None, **kwargs):
        """ Plot the light curve of a target.
    
        If `index` is None, a random index will be used. If `bands` is None,
        the target's observed band will be used.
    
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot the light curve. If None, a new figure and axes will be created.

        fig : matplotlib.figure.Figure, optional
            The figure on which to plot the light curve. If None, a new figure will be created.

        index : int, optional
            The index of the target whose light curve is to be plotted. If None, a random index is chosen.

        zp : float, optional
            Zero point magnitude for flux conversion. Default is 25.

        lc_prop : dict, optional
            Additional properties to pass to the light curve plotting function (kwargs).

        bands : list of str, optional
            The bands to plot. If None, all observed bands for the target will be used.

        show_truth : bool, optional
            Whether to show the true light curve. Default is True.

        format_time : bool, optional
            Whether to format the time axis as dates. Default is True.

        t0_format : str, optional
            The format of the reference time. Default is "mjd".

        phase_window : array-like, optional
            The phase window to plot. If None, the entire light curve will be plotted.

        **kwargs : dict
            Additional keyword arguments to pass to the plotting functions.
    
        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the light curve plot.
        """
        from matplotlib.colors import to_rgba
        from .config import get_band_color
        
        if format_time:
            from astropy.time import Time
            
        if index is None:
            index = np.random.choice(self.obs_index)

        # Data
        obs_ = self.get_target_lightcurve(index).copy()
        if phase_window is not None:
            t0 = self.targets.data["t0"].loc[index]
            phase_window = np.asarray(phase_window)+t0
            obs_ = obs_[obs_["mjd"].astype("float").between(*phase_window)]

        coef = 10 ** (-(obs_["zp"] - zp) / 2.5)
        obs_["flux_zp"] = obs_["flux"] * coef
        obs_["fluxerr_zp"] = obs_["fluxerr"] * coef

        # Model
        if bands is None:
            bands = np.unique(obs_["band"])

        # = axes and figure = #
        if ax is None:
            if fig is None:
                import matplotlib.pyplot as plt
                fig = plt.figure(figsize=[7,4])
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure
        
        colors = get_band_color(bands)
        if show_truth:
            fig = self.targets.show_lightcurve(bands, ax=ax, fig=fig, index=index, 
                                            format_time=format_time, t0_format=t0_format, 
                                            zp=zp, colors=colors,
                                            zorder=2, 
                                            **lc_prop)
        elif format_time:
            from matplotlib import dates as mdates        
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
        else:
            ax.set_xlabel("time [in day]", fontsize="large")


        # loop over bands
        for band_, color_ in zip(bands, colors):
            if color_ is None:
                ecolor = to_rgba("0.4", 0.2)
            else:
                ecolor = to_rgba(color_, 0.2)
                
            obs_band = obs_[obs_["band"] == band_]
            times = obs_band["mjd"] if not format_time else Time(obs_band["mjd"], format=t0_format).datetime
            ax.scatter(times, obs_band["flux_zp"],
                       color=color_, zorder=4, **kwargs)
            ax.errorbar(times, obs_band["flux_zp"],
                        yerr= obs_band["fluxerr_zp"],
                        ls="None", marker="None", ecolor=ecolor, 
                        zorder=3,
                        **kwargs)

        return fig
    
    # -------------- #
    #    Statics     #
    # -------------- #
    @classmethod
    def realize_survey_target_lcs(cls, targets, survey, 
                                  template_prop={}, nfirst=None,
                                  incl_error=True,
                                  client=None, discard_bands=False,
                                  trim_observations=False,
                                  phase_range=None):
        """Create the lightcurve of the input targets as they would be observed by the survey.

        These are split per survey fields.
        
        Parameters
        ----------
        targets: skysurvey.Target, skysurvey.TargetCollection
            Target data corresponding to the true target parameters
            (as given by nature).
        survey: skysurvey.Survey (or child of)
            Sky observation (what was observed when with which situation).
        template_prop: dict, optional
            kwargs for template.get(), setting the template parameters.
        nfirst: int, optional
            If given, only the first nfirst entries will be considered.
            This is a debug / test tool.
        incl_error: bool, optional
            Include error in the lightcurve.
        client: dask.distributed.Client(), optional
            Dask client to use if any. This is used for
            parallelization of the lc generation per field.
        discard_bands: bool, optional
            If True, discard bands that are not in the survey.
        trim_observations: bool, optional
            If True, trim observations to the phase range.
        phase_range: list, optional
            Phase range to consider.
            
        Returns
        -------
        list, list
            - list of dataframe (1 per fields, all targets of the field in)
            - list of fields (fieldid)
            
        See also
        --------
        from_targets_and_survey: loads the instance given targets and a survey
        """
        from .template import TemplateCollection
        if type(targets) in [list, tuple]:
            from .target.collection import TargetCollection
            targets = TargetCollection(targets) # correct format
        
        if hasattr(targets.template, "__iter__") or type(targets.template) is TemplateCollection : # collection of single-kind
            samekind_targets = targets.as_targets()
            outs = [cls._realize_survey_kindtarget_lcs(t_, survey) for t_ in samekind_targets]
            
            # lightcurves and fields
            lc_out = [l_ for l,v in outs for l_ in l if l is not None] # list of list of dataframe
            fieldids_indexes = np.vstack([np.vstack(v) for l,v in outs if v is not None]) # all fields for all cases
            fieldids_indexes = np.squeeze(fieldids_indexes) # 1-d or n-d ?
            if len(survey.fieldids.names) > 1: # multi-index
                fieldids_indexes = pandas.MultiIndex.from_arrays(fieldids_indexes.T, names=survey.fieldids.names)
            else: # single-index
                fieldids_indexes = pandas.Index(data=fieldids_indexes, name=survey.fieldids.names[0])
                
        else: # input is a single-kind target
            lc_out, fieldids_indexes = cls._realize_survey_kindtarget_lcs(targets, survey,
                                                          template_prop=template_prop,
                                                          nfirst=nfirst,
                                                          client=client,
                                                          incl_error=incl_error,
                                                          discard_bands=discard_bands,
                                                          trim_observations=trim_observations,
                                                          phase_range=phase_range)
        return lc_out, fieldids_indexes

        
    @staticmethod
    def _realize_survey_kindtarget_lcs( targets, survey,
                                           template_prop={}, nfirst=None,
                                           incl_error=True,
                                           client=None, discard_bands=False,
                                            trim_observations=False, phase_range=None):
        """Create the lightcurve of the input single-kind targets as they would be observed by the survey.

        These are split per survey fields.
        
        Parameters
        ----------
        targets: skysurvey.Target, skysurvey.TargetCollection
            Target data corresponding to the true target parameters
            (as given by nature).
        survey: skysurvey.Survey (or child of)
            Sky observation (what was observed when with which situation).
        template_prop: dict, optional
            kwargs for template.get(), setting the template parameters.
        nfirst: int, optional
            If given, only the first nfirst entries will be considered.
            This is a debug / test tool.
        incl_error: bool, optional
            Include error in the lightcurve.
        client: dask.distributed.Client(), optional
            Dask client to use if any. This is used for
            parallelization of the lc generation per field.
        discard_bands: bool, optional
            If True, discard bands that are not in the survey.
        trim_observations: bool, optional
            If True, trim observations to the phase range.
        phase_range: list, optional
            Phase range to consider.
            
        Returns
        -------
        list, list
            - list of dataframe (1 per fields, all targets of the field in)
            - list of fields (fieldid)
            
        See also
        --------
        from_targets_and_survey: loads the instance given targets and a survey
        """
        # name of template parameters
        template_columns = targets.get_template_columns()

        # fields in which target fall into
        dfieldids_ = survey.radec_to_fieldid( targets.data[["ra","dec"]] )
        

        # now let's join survey and target dataframes.
        # => first let's grab the index name, 'index by default'
        _data_index = targets.data.index.name
        if _data_index is None:
            _data_index = "index"

        # make sure index of dfieldids_ corresponds to the input one.
        dfieldids_.index.name = _data_index
        
        # merge target dataframe with matching fields.
        # note: pandas.merge conserves dtypes of fieldids, not pandas.join
        targets_data = targets.data.merge(dfieldids_, left_index=True, right_index=True)

        # ------------- #
        
        # nothing observed. Nothing to do.
        if len(targets_data) == 0: 
            return None, None

        # add fieldids (could be multi-index) into the dataframe indexing.
        target_indexed = targets_data.reset_index().set_index([_data_index]+survey.fieldids.names)

        # group targets per fieldids as realize_lightcurve will be made per fields.
        # note: best performance when passing by one groupby calls rather than indexing and xs()
        survey_data = survey.data[["mjd","band","skynoise","gain", "zp"]+survey.fieldids.names].copy()
        if survey_data.index.name is None:
            survey_data.index.name = "index_obs"
            
        gsurvey_indexed = survey_data.groupby(survey.fieldids.names)

        # get list of 
        # This works for any size of fieldids
        names = survey.fieldids.names
        levels = np.arange(1, len(names)+1).tolist()
        if len(levels)==1:
            levels = levels[0] # single index dataframe

        fieldids_indexes = target_indexed.groupby(level=levels).size().index
        if nfirst is not None:
            fieldids_indexes = fieldids_indexes[:nfirst]
            
        # ------------- #
        
        # Build a LC for a given index
        sncosmo_model_base = targets.get_template(as_model=True, **template_prop)            
        def _get_index_lc_input_(index_):
            """Get the lightcurve input for a given index."""
            try:
                this_survey = gsurvey_indexed.get_group(index_).copy()
                #survey_indexed.xs(index_)[["mjd","band","skynoise","gain", "zp"]]
            except:
                # no observations for this fieldids 
                return None

            # get() returns copy
            sncosmo_model = copy( sncosmo_model_base )
            # survey matching the input fieldids row

            # Taking the data we need
            this_target = target_indexed.xs(index_, level=levels)[template_columns].copy()
            return sncosmo_model, this_survey, this_target

        data = [_get_index_lc_input_(index_) for index_ in fieldids_indexes]
        lc_out = [_get_obsdata_(data_, incl_error=incl_error,
                                    discard_bands=discard_bands,
                                    trim_observations=trim_observations,
                                    phase_range=phase_range)
                          for data_ in data if data_ is not None]

        return lc_out, fieldids_indexes

    # ============== #
    #   Properties   # 
    # ============== #
    @property
    def data(self):
        """Lightcurve data as observed by the survey."""
        return self._data

    @property
    def _data_index(self):
        """Name of data index."""
        if not hasattr(self, "_hdata_index"):
            self._hdata_index = "index"
        return self._hdata_index
    
    @property
    def targets(self):
        """Target data corresponding to the true target parameters."""
        return self._targets

    @property
    def survey(self):
        """Survey that has been used to generate the dataset."""
        return self._survey

    @property
    def obs_index(self):
        """Index of the observed target."""
        if not hasattr(self,"_obs_index") or self._obs_index is None:
            self._obs_index = self.data.index.get_level_values(0).unique().sort_values()
            
        return self._obs_index
