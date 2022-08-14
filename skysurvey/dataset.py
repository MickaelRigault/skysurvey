

""" This library concerns the data as observed """

#
import pandas
import numpy as np
#
import sncosmo
from astropy.table import Table


__all__ = ["DataSet"]


def get_obsdata(template, observations, parameters, zp=25, zpsys="ab"):
    """ """
    # observation of that field    
    observations[["zp","zpsys"]] = [zp, zpsys]
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
        per_fieldid = cls._realize_lc_perfieldid_from_survey_and_target(targets, survey, 
                                                                        use_dask=use_dask,
                                                                        inplace=targetsdata_inplace)
        if use_dask:
            if client is None:
                import dask
                all_outs = dask.delayed(list)(per_fieldid).compute()
            else:
                f_all_outs = client.compute(all_outs)
                all_outs = client.gather(f_all_outs)
            
        data = pandas.concat(all_outs)
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
        
    def set_targets(self, targets):
        """ observation lightcurve data """
        self._targets = targets

    def set_survey(self, survey):
        """ observation lightcurve data """
        self._survey = survey

    # -------- #
    #  GETTER  #
    # -------- #

    
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
            
        all_out = []
        for fieldid_ in targets_data["fieldid"].unique():
            # What kind of template ?
            template = targets.get_template() # in the loop to avoid dask conflict, to be checked
            
            # get the given field observation
            this_survey = survey.data[survey.data["fieldid"] == fieldid_][["mjd","band","skynoise","gain"]]

            # taking the data we need
            existing_columns = np.asarray(template.param_names)[np.in1d(template.param_names, targets_data.columns)]
            this_target = targets_data[targets_data["fieldid"] == fieldid_][existing_columns]
            
            # realize the lightcurve for this fieldid
            if use_dask:
                this_out = dask.delayed(get_obsdata)(template, this_survey, this_target)
            else:
                this_out = get_obsdata(template, this_survey, this_target)
            
            all_out.append(this_out)
        
        return all_out
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
        
