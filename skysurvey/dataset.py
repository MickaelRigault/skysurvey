

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
    list_of_observations = sncosmo.realize_lcs(sncosmo_obs,template, list_of_parameters)
    if len(list_of_observations) == 0:
        return None
    
    return pandas.concat([l.to_pandas() for l in list_of_observations],  keys=parameters.index)



# ================== #
#                    #
#    DataSet         #
#                    #
# ================== #
class DataSet( object ):
    
    def __init__(self, data, targets=None, survey=None):
        """ The DataSet should most likely be created from 
        classmethod such as:
        - DataSet.from_targets_and_survey( yourtargets, yoursurvey)
        
        - DataSet.read_parquet( parquet_file)

        - DataSet.read_from_directory()

        """
        self.set_data(data)
        self.set_targets(targets)
        self.set_survey(survey)
        
    @classmethod
    def from_targets_and_survey(cls, targets, survey, 
                                use_dask=False, client=None, 
                                targetsdata_inplace=False):
        """ """
        fieldids, per_fieldid = cls._realize_lc_perfieldid_from_survey_and_target(targets, survey, 
                                                                        use_dask=use_dask,
                                                                        inplace=targetsdata_inplace)
        if use_dask:
            if client is None:
                import dask
                all_outs = dask.delayed(list)(per_fieldid).compute()
            else:
                f_all_outs = client.compute(all_outs)
                all_outs = client.gather(f_all_outs)
        else:
            all_outs = per_fieldid

        # keep track of why field is what.
        data = pandas.concat(all_outs, keys=fieldids).reset_index(level=0).rename({"level_0":"fieldid"}, axis=1)
        return cls(data, targets=targets, survey=survey)

    @classmethod
    def read_parquet(cls, parquetfile, survey=None, targets=None, **kwargs):
        """ Loads a stored dataset. Only the observation data can be loaded this way, 
        not the survey nor the targets (truth). """
        data = pandas.read_parquet(parquetfile, **kwargs)
        return cls(data, survey=survey, targets=targets)

    @classmethod
    def read_from_directory(cls, dirname, **kwargs):
        """ """
        raise NotImplementedError("read_from_directory is not yet available.")
        
    # ============== #
    #   Method       #
    # ============== #
    # -------- #
    #  SETTER  #
    # -------- #
    def set_data(self, data):
        """ observation lightcurve data """
        self._data = data
        self._obs_index = None
        
    def set_targets(self, targets):
        """ observation lightcurve data """
        self._targets = targets

    def set_survey(self, survey):
        """ observation lightcurve data """
        self._survey = survey

    # -------- #
    #  GETTER  #
    # -------- #
    def get_ndetection(self, detlimit=5, per_band=False):
        """ get the number of detection (flux/fluxerr)>detlimit per observed target (and per_band if per_band=True). """
        data = self.data.copy()
        data["detected"] = (data["flux"]/data["fluxerr"])>detlimit
        if per_band:
            ndetection = data.reset_index().set_index(["level_0","level_1","band"]).groupby(level=[0,2])["detected"].sum()
        else:
            ndetection = data.groupby(level=0)["detected"].sum()

            
        return ndetection

    
    # -------- #
    #  FIT     #
    # -------- #
    def fit_lightcurve(self, source, index=None, use_dask=True,
                        incl_dust=True, 
                        phase_fitrange=[-50,200],
                        fixedparams = None,


                        
                        guessparams = None,
                        bounds = None,
                        add_truth=True,
                        **kwargs):
        """ 
        phase_fitrange can only work if t0 is given in fixed or guess

        add_truth: bool
            if self.targets.data exists and add_truth=True, the colunm "truth" will be 
            added to the results getting values from self.targets.data.
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
            res = dask.delayed(np.asarray)(results, dtype="object").compute()
            flatres = np.ravel(res)
            results = pandas.concat(flatres[::2], keys=index)
            metas = pandas.concat(flatres[1::2], keys=index)
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
                                lc_prop={}, bands=None, 
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
        obs_ = self.data.xs(index).copy()
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

        colors = get_band_color(bands)
        fig = self.targets.show_lightcurve(bands, ax=ax, fig=fig, index=index, 
                                           format_time=format_time, t0_format=t0_format, 
                                           zp=zp, colors=colors,
                                           zorder=2, 
                                           **lc_prop)
        ax = fig.axes[0]



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
    @staticmethod
    def _realize_lc_perfieldid_from_survey_and_target(targets, survey, template_source=None,
                                                      use_dask=False, inplace=False, template_prop={}):
        """ """
        if use_dask:
            import dask

        if template_source is None:
            template_source = targets.template.source
            
        targets_data = targets.data.copy() if not inplace else targets.data
        targets_data["fieldid"] = survey.radec_to_fieldid(*targets_data[["ra","dec"]].values.T)

        targets_data = targets_data.explode("fieldid")
        all_out = []

        fieldids = targets_data["fieldid"].unique()
        for fieldid_ in fieldids:
            # What kind of template ?
            # in the loop to avoid dask conflict, to be checked
            template = Template._get(template_source, **template_prop) 
            
            # get the given field observation
            this_survey = survey.data[survey.data["fieldid"] == fieldid_][["mjd","band","skynoise","gain", "zp"]]

            # taking the data we need
            existing_columns = np.asarray(template.param_names)[np.in1d(template.param_names, targets_data.columns)]
            this_target = targets_data[targets_data["fieldid"] == fieldid_][existing_columns]
            
            # realize the lightcurve for this fieldid
            if use_dask:
                this_out = dask.delayed(get_obsdata)(template, this_survey, this_target)
            else:
                this_out = get_obsdata(template, this_survey, this_target)
            
            all_out.append(this_out)
        
        return fieldids, all_out
    
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
