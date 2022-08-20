

""" This library concerns the data as observed """

#
import pandas
import numpy as np
#
import sncosmo
from astropy.table import Table


__all__ = ["DataSet"]


def get_obsdata(template, observations, parameters, zpsys="ab"):
    """ """
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
                                use_dask=True, client=None, 
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
    def _realize_lc_perfieldid_from_survey_and_target(targets, survey, 
                                                      use_dask=True, inplace=False):
        """ """
        if use_dask:
            import dask
            
        targets_data = targets.data.copy() if not inplace else targets.data
        targets_data["fieldid"] = survey.radec_to_fieldid(*targets_data[["ra","dec"]].values.T)

        targets_data = targets_data.explode("fieldid")
        all_out = []

        fieldids = targets_data["fieldid"].unique()
        for fieldid_ in fieldids:
            # What kind of template ?
            template = targets.get_template() # in the loop to avoid dask conflict, to be checked
            
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
