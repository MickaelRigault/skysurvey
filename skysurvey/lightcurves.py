""" module to create lightcurves (used by dataset.realize_lightcurves) """


import copy
#
import pandas
import numpy as np
#
import sncosmo
from astropy.table import Table


def get_obsdata(template, observations, parameters, zpsys="ab", incl_error=True, discard_bands=False,
                trim_observations=False, phase_range=None):
    """ get observed data using ``sncosmo.realize_lcs()``

    Parameters
    ----------
    template: sncosmo.Model
        an sncosmo model from which we can draw observations
        
    observations: pandas.DataFrame
        Dataframe containing the observing infortation.
        requested entries: TBD
    
    parameters: pandas.DataFrame
        Dataframe containing the target parameters information.
        These depend on you model. 

    incl_error: bool
        should the returned flux contain a random gaussian scatter
        drawn from the flux_err ?
        If False, lightcurve flux are "perfect".

    discard_bands: bool    
        If True, if the sncosmo model is not defined in a given observeing band, the observation is discarded altogether, 
        to prevent sncosmo.realize_lcs() from crashing. This only works for bands that are too blue for now.

    Returns
    -------
    MultiIndex DataFrame
        all the observations for all targets

    See also
    --------
    DataSet.from_targets_and_survey: generate a DataSet from target and survey's object
    """
    # if missing, we assume to work in a 'zpsys' system.
    if "zpsys" not in observations:
        observations["zpsys"] = zpsys

    dataobs = observations[["band", "mjd", "zp", "zpsys", "gain", "skynoise"]]
    
    # should this discard band not covered by sncosmo ?
    if not discard_bands:        
        # realize lightcurve
        list_of_observations = realize_lightcurves(dataobs, template, parameters,
                                                   scatter=incl_error,
                                                   trim_observations=trim_observations,
                                                    phase_range=phase_range)
        return list_of_observations # could be None

    else:
        # we create each lightcurves per bands.
        bands = np.unique(observations['band'])
        lcs = []
        for band in bands:
            masked_dataobs = dataobs[dataobs['band']==band]
            z_max = sncosmo.get_bandpass(band).minwave()/template.minwave()-1
            masked_parameters = parameters[parameters['z'] < z_max]
            list_of_observations = realize_lightcurves(masked_dataobs,
                                                       template,
                                                       masked_parameters,
                                                       scatter=incl_error,
                                                       trim_observations=trim_observations,
                                                       phase_range=phase_range)
            
            if list_of_observations is not None and len(list_of_observations) > 0:
                lcs.append( list_of_observations )
            
        return pandas.concat(lcs)
    
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



# ====================== #
#                        #
#  Realize Lightcurves   #
#                        #
# ====================== #
def realize_lightcurves(observations, model, parameters,
                        trim_observations=False, phase_range=None,
                        scatter=True):
    """Realize data for a set of SNe given a set of observations.

    Note: adapted from sncosmo.realize_lcs, but:
         - replacing astropy.Table by pandas.DataFrame
         - removing ability to use aliases. (no time lost in that)
         - removing thresh (no time lost in that)

    Parameters
    ----------
    observations : `pandas.DataFrame`
        Table of observations. Must contain the following column names:
        ``band``, ``mjd``, ``zp``, ``zpsys``, ``gain``, ``skynoise``.
    model : `sncosmo.Model`
        The model to use in the simulation.
    parameters : pandas.DataFrame
        List of parameters to feed to the model for realizing each light curve.
    trim_observations : bool, optional
        If True, only observations with times between
        ``model.mintime()`` and ``model.maxtime()`` are included in
        result table for each SN. Default is False.
    phase_range: list, None, optional
        If given, only observations within the given rest-frame phase range
        will be considered.
    scatter : bool, optional
        If True, the ``flux`` value of the realized data is calculated by
        adding  a random number drawn from a Normal Distribution with a
        standard deviation equal to the ``fluxerror`` of the observation to
        the bandflux value of the observation calculated from model. Default
        is True.

    Returns
    -------
    sne : list of `pandas.DataFrame`
        Table of realized data for each item in ``params``.

    Notes
    -----
    ``skynoise`` is the image background contribution to the flux measurement
    error (in units corresponding to the specified zeropoint and zeropoint
    system). To get the error on a given measurement, ``skynoise`` is added
    in quadrature to the photon noise from the source.

    It is left up to the user to calculate ``skynoise`` as they see fit as the
    details depend on how photometry is done and possibly how the PSF is
    is modeled. As a simple example, assuming a Gaussian PSF, and perfect
    PSF photometry, ``skynoise`` would be ``4 * pi * sigma_PSF * sigma_pixel``
    where ``sigma_PSF`` is the standard deviation of the PSF in pixels and
    ``sigma_pixel`` is the background noise in a single pixel in counts.
    """
    lcs = []
    indexes = []

    # Copy model so we don't mess up the user's model.
    model = copy.copy(model)

    # sn parameters
    list_of_parameters = [p_.to_dict() for i_,p_ in parameters.iterrows()] # sncosmo format

    # loops over targets
    for target_index, param in parameters.iterrows():
        
        # update the model to the current parameters
        model.set( **param.to_dict() )

        # Select times for output that fall within tmin amd tmax of the model
        if trim_observations: # {min/max}time includes redshift phase dilatation.
            snobs = observations[ observations['mjd'].between(model.mintime(), model.maxtime()) ]
        else:
            snobs = observations

        if phase_range is not None: # copy made before
            phase_range_obsframe = np.asarray( phase_range ) * (1 + model.get("z")) + model.get("t0")
            snobs = observations[ observations['mjd'].between(*phase_range_obsframe) ]

        # explicitly detect no observations and add an empty table
        if len(snobs) == 0:
            continue

        flux = model.bandflux(snobs['band'],
                              snobs['mjd'],
                              zp=snobs['zp'],
                              zpsys=snobs['zpsys'])

        fluxerr = np.sqrt(snobs['skynoise']**2 + np.abs(flux) / snobs['gain'])
    
        # Scatter fluxes by the fluxerr
        # np.atleast_1d is necessary here because of an apparent bug in
        # np.random.normal: when the inputs are both length 1 arrays,
        # the output is a Python float!
        if scatter:
            flux = np.atleast_1d( np.random.normal(flux, fluxerr) )
            
        # output
        data = snobs.merge(pandas.DataFrame({"flux": flux, "fluxerr": fluxerr}, index=snobs.index),
                                 left_index=True, right_index=True)
                            
        lcs.append( data )
        indexes.append( target_index )

    if len(lcs)==0:
        return None
    
    return pandas.concat(lcs, keys=pandas.Index(indexes, name=parameters.index.name) )
