

""" This library concerns the data as observed """

#
import pandas
import numpy as np
#
import sncosmo
from astropy.table import Table

from .template import Template


__all__ = ["DataSet", "get_obsdata"]

def get_obsdata(template, observations, parameters, zpsys="ab"):
    """ get observed data using ``sncosmo.realize_lcs()``

    Parameters
    ----------
    template: sncosmo.Model
        an sncosmo model from which we can draw observations
        (passed to 
        
    observations: pandas.DataFrame
        Dataframe containing the observing infortation.
        requested entries: TBD
    
    parameters: pandas.DataFrame
        Dataframe containing the target parameters information.
        These depend on you model. 

    Returns
    -------
    MultiIndex DataFrame
        all the observations for all targets

    See also
    --------
    DataSet.from_targets_and_survey: generate a DataSet from target and survey's object
    """
    # observation of that field
    if "zpsys" not in observations:
        observations["zpsys"] = zpsys
        
    sncosmo_obs = Table.from_pandas(observations.rename({"mjd":"time"}, axis=1)) # sncosmo format
    
    # sn parameters
    list_of_parameters = [p_.to_dict() for i_,p_ in parameters.iterrows()] # sncosmo format
    
    # realize LC
    list_of_observations = sncosmo.realize_lcs(sncosmo_obs, template, list_of_parameters)
    if len(list_of_observations) == 0:
        return None
    
    return pandas.concat([l.to_pandas().set_index(observations.index) for l in list_of_observations],  keys=parameters.index)

def _get_obsdata_(data, **kwargs):
    """ internal method to simplify get_obsdata using single input (for map)

    Parameters
    ----------
    data: list
        3 entries:
        template: sncosmo.Model
            an sncosmo model from which we can draw observations
            
        observations: pandas.DataFrame
            Dataframe containing the observing infortation.
            requested entries: TBD
    
        parameters: pandas.DataFrame
            Dataframe containing the target parameters information.
            These depend on you model. 

    **kwargs goes to get_obsdata

    Returns
    -------
    MultiIndex DataFrame
        all the observations for all targets

    See also
    --------
    DataSet.from_targets_and_survey: generate a DataSet from target and survey's object
    """
    return get_obsdata(*data, **kwargs)


# ================== #
#                    #
#    DataSet         #
#                    #
# ================== #
class DataSet( object ):
    
    def __init__(self, data, targets=None, survey=None):
        """ 

        See also
        --------
        from_targets_and_survey: loads a dataset (observed data) given targets and survey
        read_parquet: loads a stored dataset
        """
        self.set_data(data)
        self.set_targets(targets)
        self.set_survey(survey)
        
    @classmethod
    def from_targets_and_survey(cls, targets, survey, template=None, client=None, **kwargs):
        """ loads a dataset (observed data) given targets and a survey

        This first matches the targets (given targets.data[["ra","dec"]]) with the
        survey to find which target has been observed with which field.
        Then simulate the targets lightcurves given the observing data (survey.data).
        

        Parameters
        ----------
        targets: skysurvey.Target (or child of)
            target data corresponding to the true target parameters  
            (as given by nature)
            
        survey: skysurvey.Survey (or child of)
            sky observation (what was observed when with which situation)

        client: dask.distributed.Client()
            dask client to use if any. This is used for 
            parallelization of the lc generation per field.

        **kwargs goes to realize_survey_target_lcs

        Returns
        -------
        class instance
            the observation data have been derived and stored as self.data

        See also
        --------
        read_parquet: loads a stored dataset
        """
        lightcurves, fieldids = cls.realize_survey_target_lcs(targets, survey, template=template,
                                                                  client=client,
                                                 **kwargs)
        data = pandas.concat(lightcurves, keys=fieldids # store fieldid
                                 ).reset_index(survey.fieldids.names)
        return cls(data, targets=targets, survey=survey)

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
    def get_data(self, detected=None, detlimit=5, 
                     phase_range=None, t0=None):
        """ """
        data = self.data.copy()
        if phase_range is not None:
            if t0 is None:
                t0 = self.targets.data["t0"].copy()
            
            data["phase"] = data["time"] - t0.reindex(data.index, level=0)
            data = data[data["phase"].between(*phase_range)]

        if detected is not None:
            data["detection"] = data["flux"]/data["fluxerr"]
            if detected:
                data = data[ data["detection"] ] > detlimit
            else:
                data = data[ data["detection"] ] <= detlimit

        return data
        
    def get_ndetection(self, detlimit=5, per_band=False):
        """ get the number of detection for each lightcurves

        Basically computes the number of datapoints with (flux/fluxerr)>detlimit)

        Parameters
        ----------
        detlimit: float, int
            detection limit below which a point is not considered
            (cut a > not >=)
        
        per_band: bool
            should be computation be made per band ?
            if true it will then be per target *and* per band.

        Returns
        -------
        pandas.Series
            the number of detected point per target (and per band if per_band=True)
        """
        data = self.data.copy()
        data["detected"] = (data["flux"]/data["fluxerr"])>detlimit
        if per_band:
            groupby = ["index","band"]
        else:
            groupby = "index"
        
        ndetection = data.groupby(groupby)["detected"].sum()
        return ndetection

    def get_target_lightcurve(self, index, detection_only=False, detlimit=5.):
        """ get the observation of the given target.
        
        = short cut to self.data.xs(index) = 

        Parameters
        ----------
        detection_only: bool
            only the detected points of the lightcurves 

        detlimit: float
            definition of a detection (flux/fluxerr>=detlimit)

        Returns
        -------
        pandas.DataFrame
            the lightcurve

        """
        data = self.data.xs(index)
        if detection_only:
            data = data[data["flux"]/data["fluxerr"]>=detlimit]

        return data
    
    # -------- #
    #  FIT     #
    # -------- #
    def fit_lightcurves(self, source, index=None,
                           use_dask=True,
                           phase_fitrange=[-50,200],
                           fixedparams = None,
                           guessparams = None,
                           bounds = None,
                           incl_dust=True, 
                           add_truth=True,
                           **kwargs):
        """ fit the template source model to the observed data.

        Basically loops over the targets an use sncosmo.fit_lc()

        Parameters
        ----------
        source: str
            name of the template ( will use ``sncosmo.Model(source)``)
            
        index: list
            select the target to be fitted using their index

        use_dask: bool
            shall this use dask to distribute the computation ?

        phase_fitrange: 2d-array, None
            if not None, only the given phase_range will be considered
            for the fit.  t0 is taken from fixedparams or from guessparams, 
            whichever comes first.
            
        fixedparams: dict
            fix parameters to this value for the fit.
            a parameter could be fixed at a given value: e.g. {"mw_ebv":0}
            or it must be one per fitted target: e.g. {"z":dataset.targets.data["z"]}
            
        fixedparams: dict
            guess parameters to this value for the fit.
            a parameter could be fixed at a given value: e.g. {"mw_ebv":0}
            or it must be one per fitted target: e.g. {"z":dataset.targets.data["z"]}
            
        bounds: dict
            boundaries for the parameters.
            (see example)

        add_truth: bool
            = ignored if self.targets is not set =
            should the true parameters be added to the results ("thruth" columns)

        incl_dust: bool
            should the template include the Milky Way dust extinction
            parameters ?


        Returns
        -------
        pandas.DataFrame, pandas.DataFrame
            multi-index dataframe of the fits results, errors and covariances
            and the pandas.DataFrame fit meta data.

        
        Examples
        --------
        fit the lightcurves of targets having at least 5 detection, fixing the redshift and the mw parameters
        the t0 is only a guess (but actually guessed at the truth here)

        >>> detstat = dataset.get_ndetection()
        >>> detected = detstat[detstat>5].index
        >>> results, meta = dataset.fit_lightcurve("salt2", 
                                                  use_dask=True, index=detected
                                                  phase_fitrange=[-30,60],
                                                  add_truth=True,
                                                  fixedparams={"z":dataset.targets.data.loc[detected]["z"],
                                                               "mwr_v":3.1, "mwebv":0},
                                                  guessparams={"t0":dataset.targets.data.loc[detected]["t0"]},
                                                  bounds={"t0":dataset.targets.data.loc[detected]["t0"].apply(lambda x: [x-5, x+5])}
                                                 )
        """
        if use_dask:
            import dask

        if index is None:
            index = self.obs_index.values

        if phase_fitrange is not None:
            phase_fitrange = np.asarray(phase_fitrange)

        def _format_paramin_(paramin):
            """ """
            if type(paramin) is dict:
                # most flexible format found
                temp_ = pandas.DataFrame(index=index)
                for k,v in paramin.items(): 
                    temp_[k] = v
                paramin = temp_.copy()

            return paramin

        fixedparams = _format_paramin_(fixedparams)
        guessparams = _format_paramin_(guessparams)
        bounds = _format_paramin_(bounds)

        results = []
        metas = []

        
        for i in index:
            if use_dask:
                template = dask.delayed(Template)(source)
            else:
                template = Template(source)
                
            # Data
            data_to_fit = self.data.xs(i)
            #
            fixed_ = fixedparams.loc[i].to_dict() if fixedparams is not None else None
            guess_ = guessparams.loc[i].to_dict() if guessparams is not None else None
            bounds_ = bounds.loc[i].to_dict() if bounds is not None else None
            # - t0 for datarange
            if phase_fitrange is not None:
                t0 = fixed_.get("t0", guess_.get("t0", None)) # from fixed or from guess or None
                if t0 is not None:
                    data_to_fit = data_to_fit[data_to_fit["time"].between(*(t0+phase_fitrange))]

            prop = {**dict(fixedparams=fixed_, guessparams=guess_,
                           bounds=bounds_), 
                    **kwargs}

            if use_dask:
                # is already delayed
                result_meta = template.fit_data(data_to_fit,  **prop) # this create a new sncosmo_model inside fit_data
                results.append(result_meta)

            else:
                result, meta = template.fit_data(data_to_fit,  **prop)
                results.append(result)
                metas.append(meta)

        if use_dask:
            res = dask.delayed(list)(results).compute()
            res_, meta_ = np.array(res, dtype="object").T
            results = pandas.concat(res_, keys=index)
            metas = pandas.concat(meta_, keys=index)
        else:
            results = pandas.concat(results, keys=index)
            metas = pandas.concat(metas, keys=index)

        if add_truth and self.targets is not None:
            truth = self.targets.data.loc[index].stack()

            truth.name = "truth"
            results = results.merge(truth, left_index=True, right_index=True)
            
        return results, metas    
    # -------- #
    #  PLOTTER #
    # -------- #
    def show_target_lightcurve(self, ax=None, fig=None, index=None, zp=25,
                                lc_prop={}, bands=None, show_truth=True,
                                format_time=True, t0_format="mjd", 
                                phase_window=None, **kwargs):
        """ if index is None, a random index will be used. 
        if bands is None, the target's observed band will be used.
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
            obs_ = obs_[obs_["time"].between(*phase_window)]

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



        for band_, color_ in zip(bands, colors):
            obs_band = obs_[obs_["band"] == band_]
            times = obs_band["time"] if not format_time else Time(obs_band["time"], format=t0_format).datetime
            ax.scatter(times, obs_band["flux_zp"],
                       color=color_, zorder=4, **kwargs)
            ax.errorbar(times, obs_band["flux_zp"],
                        yerr= obs_band["fluxerr_zp"],
                        ls="None", marker="None", ecolor=to_rgba(color_, 0.2), 
                        zorder=3,
                        **kwargs)

        return fig
    
    # -------------- #
    #    Statics     #
    # -------------- #
    @classmethod
    def realize_survey_target_lcs(cls, targets, survey, template=None,
                                  template_prop={}, nfirst=None,
                                  client=None):
        """ creates the lightcurve of the input targets as they 
        would be observed by the survey. 
        = These are split per survey fields. =
        
        
        Parameters
        ----------
        targets: skysurvey.Target, skysurvey.TargetCollection
            target data corresponding to the true target parameters  
            (as given by nature)
            
        survey: skysurvey.Survey (or child of)
            sky observation (what was observed when with which situation)

        template: Template
            template to use to generate the lightcurve.
            If None given, the target's template is used.

        template_prop: dict
            kwargs for template.get(), setting the template parameters
            
        nfirst: int
            if given, only the first nfrist entries will be considered.
            This is a debug / test tool.
        
        client: dask.distributed.Client()
            dask client to use if any. This is used for 
            parallelization of the lc generation per field.
        
            
        Returns
        -------
        list, list
            - list of dataframe (1 per fields, all targets of the field in)
            - list of fields (fieldid)
            
        See also
        --------
        from_targets_and_survey: laods the instance given targets and a survey
        """
        if hasattr(targets.template, "__iter__"): # collection of single-kind
            samekind_targets = targets.as_targets()
            outs = [cls._realize_survey_kindtarget_lcs(t_, survey)
                        for t_ in samekind_targets]
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
                                                          template=template,
                                                          template_prop=template_prop,
                                                          nfirst=nfirst,
                                                          client=client)
        return lc_out, fieldids_indexes

        
    @staticmethod
    def _realize_survey_kindtarget_lcs( targets, survey, template=None,
                                           template_prop={}, nfirst=None,
                                           client=None):
        """ creates the lightcurve of the input single-kind 
        targets as they  would be observed by the survey. 
        = These are split per survey fields. =
        
        
        Parameters
        ----------
        targets: skysurvey.Target, skysurvey.TargetCollection
            target data corresponding to the true target parameters  
            (as given by nature)
            
        survey: skysurvey.Survey (or child of)
            sky observation (what was observed when with which situation)

        template: Template
            template to use to generate the lightcurve.
            If None given, the target's template is used.

        template_prop: dict
            kwargs for template.get(), setting the template parameters
            
        nfirst: int
            if given, only the first nfrist entries will be considered.
            This is a debug / test tool.
        
        client: dask.distributed.Client()
            dask client to use if any. This is used for 
            parallelization of the lc generation per field.
        
            
        Returns
        -------
        list, list
            - list of dataframe (1 per fields, all targets of the field in)
            - list of fields (fieldid)
            
        See also
        --------
        from_targets_and_survey: laods the instance given targets and a survey
        """
        
        if template is None:
            template = targets.template

        template_columns = targets.get_template_columns()

        dfieldids_ = survey.radec_to_fieldid(targets.data[["ra","dec"]])
        # merge conserves the dtypes of fieldids, not join.
        targets_data = targets.data.merge(dfieldids_, left_index=True, right_index=True)
        if len(targets_data) == 0: # no field containing this target
            return None, None

        # index them per fieldids names.
        target_indexed = targets_data.reset_index().set_index(["index"]+survey.fieldids.names)
        
        # best performance when passing by one groupby calls rather than indexing and xs()
        gsurvey_indexed = survey.data[["mjd","band","skynoise","gain", "zp"]+survey.fieldids.names
                                     ].groupby(survey.fieldids.names)

        # get list of 
        # This works for any size of fieldids
        names = survey.fieldids.names
        levels = np.arange(1, len(names)+1).tolist()
        if len(levels)==1:
            levels = levels[0] # single index dataframe

        fieldids_indexes = target_indexed.groupby(level=levels).size().index
        if nfirst is not None:
            fieldids_indexes = fieldids_indexes[:nfirst]
            
        
        # Build a LC for a given index
        def get_index_lc_input(index_):
            """ """
            try:
                this_survey = gsurvey_indexed.get_group(index_).copy()
                #survey_indexed.xs(index_)[["mjd","band","skynoise","gain", "zp"]]
            except:
                # no observations for this fieldids 
                return None

            # get() returns copy
            sncosmo_model = template.get(**template_prop)
            # survey matching the input fieldids row

            # Taking the data we need
            this_target = target_indexed.xs(index_, level=levels)[template_columns].copy()
            return sncosmo_model, this_survey, this_target

        data = [get_index_lc_input(index_) for index_ in fieldids_indexes]
        if client is not None:
            big_future = client.scatter(data) # scatter data
            futures_ = client.map(_get_obsdata_, big_future)
            lc_out = client.gather(futures_)
        else:
            lc_out = [_get_obsdata_(data_) for data_ in data if data_ is not None]

        return lc_out, fieldids_indexes

    # ============== #
    #   Properties   #
    # ============== #
    @property
    def data(self):
        """ """
        return self._data

    @property
    def targets(self):
        """ """
        return self._targets

    @property
    def survey(self):
        """ """
        return self._survey

    @property
    def obs_index(self):
        """ index of the observed target """
        if not hasattr(self,"_obs_index") or self._obs_index is None:
            self._obs_index = self.data.index.get_level_values(0).unique().sort_values()
            
        return self._obs_index
