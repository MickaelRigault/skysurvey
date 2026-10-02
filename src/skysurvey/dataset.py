"""
This module concerns the data as observed.
DataSet joins information for a Transient (list of true data) and a Survey (what has been observed when).
It generates real lightcurves observations.
"""

import numpy as np
import pandas
import sncosmo
import warnings

from .target.collection import TargetCollection

# ================== #
#                    #
#    DataSet         #
#                    #
# ================== #
class DataSet(object):
    """
    A class for managing and realistic transient light curves given true data and survey observing logs.

    This class provides methods to load, manipulate, and visualize light curve data
    based on target and survey information.

    The classmethod ``DataSet.from_targets_and_survey()`` should be favored for loading the dataset.

    Parameters
    ----------
    data : `pandas.DataFrame`
        Multi-index dataframe corresponding to the concatenation of all targets observations.

    targets : ``skysurvey.Target`` or child of, optional
        Target data corresponding to the true target parameters (as given by nature).

    survey : ``skysurvey.Survey`` or child of, optional
        Survey that has been used to generate the dataset (if known).

    See Also
    --------
    :func:`from_targets_and_survey` : Loads a dataset (observed data) given targets and survey.
    :func:`read_parquet` : Loads a stored dataset.
    """

    def __init__(self, data, targets=None, survey=None):
        """Initialize the DataSet class."""
        self.set_data(data)
        self.set_targets(targets)
        self.set_survey(survey)

    @classmethod
    def from_targets_and_survey(cls, targets, survey, incl_error=True, # client=None,
                                phase_range=[-50, +200], progress_bar=False, seed=None,
                                discard_bands=True):
        """Loads a dataset (observed data) given targets and a survey.

        This first matches the targets (given ``targets.data[['ra','dec']]``) with the
        survey to find which target has been observed with which field.
        Then simulate the targets lightcurves given the observing data (``survey.data``).


        Parameters
        ----------
        targets: ``skysurvey.Target``, list, ``skysurvey.TargetCollection``
            Target data corresponding to the true target parameters
            (as given by nature). Could be a list

        survey: ``skysurvey.Survey`` (or child of)
            Sky observation (what was observed when with which situation).

        incl_error: bool, optional
            Include error in the lightcurve.
            If False, the flux is the true model flux.

        phase_range: list, None, optional
            Rest-frame phase range to be used for simulating
            the lightcurves. If None, no cut is applied on time range for the logs.

        progress_bar: bool, optional
            shall this display a progress bar associated to the generation of targets ?
            (uses tqdm)

        seed : None, int, Generator, RandomState, optional
            = ignored if incl_error=False =
            (docstring adapted from ``np.random.default_rng``)
            If None, a fresh seed will be pulled.
            If an `int`, it will be passed to `SeedSequence` to derive the initial `BitGenerator` state.
            Additionally, when passed a `(Bit)Generator`, it will be returned unaltered.
            When passed a legacy `RandomState` instance it will be coerced to a `Generator`.

        discard_bands : bool, optional
            If True, discards the bands that includes wavelength for which the (observer-frame) target SED is not defined.
            This prevents crashing the code due to an error from `sncosmo`.

        Returns
        -------
        dataset
            instance of a `DataSet` loaded from the given targets.
        """
        from .template import Template
        lc_prop = dict(progress_bar=progress_bar,
                        incl_error=incl_error,
                        phase_range=phase_range,
                        seed=seed, discard_bands=discard_bands,
                        single_model=True)
        # single target:
        if "skysurvey" in str(type(targets)) and type(targets.template) is Template:
            lcs = cls.lightcurve_from_targets_and_survey(targets, survey, **lc_prop)
        else:
            if type(targets) in [list, tuple]:
                targets = TargetCollection(targets)

            single_model_targets = targets.as_targets()
            lcs_ = [cls.lightcurve_from_targets_and_survey(target_, survey, **lc_prop)
                    for target_ in single_model_targets]
            lcs = pandas.concat(lcs_)

        return cls(lcs, targets=targets, survey=survey)

    @classmethod
    def lightcurve_from_targets_and_survey(cls, targets, survey, progress_bar=False,
                                    incl_error=True, phase_range=[-50, 200],
                                    seed=None, discard_bands=True,
                                    single_model=True):
        """ Loads a dataset (observed data) given targets and a survey."""
        if progress_bar:
            from tqdm import tqdm

        field_names = survey.fieldids.names

        # --- 1. one merge: one row per (target, observation)
        dfieldids_ = survey.radec_to_fieldid(targets.data[["ra", "dec"]])
        dfieldids_.index.name = "index"
        tdata = targets.data[["t0", "z"]].copy()
        tdata.index.name = "index"
        tdata = tdata.merge(dfieldids_, left_index=True, right_index=True).reset_index()
        logs = survey.data[["mjd", "band", "skynoise", "gain", "zp"] + field_names]
        logs = logs.rename_axis("index_obs").reset_index()
        lc = tdata.merge(logs, on=field_names, how="inner")

        # --- 2. vectorised phase cut
        if phase_range is not None:
            # produce phase cut already now
            # to have the smallest Dataframe to carry on.
            # Phases are to be understood "rest-frame".
            lc = lc[ ( (lc["mjd"]-lc["t0"]) / (1+lc["z"]) ).between(*phase_range, inclusive='both')]

        # --- 3. sort once -> contiguous blocks
        lc = lc.sort_values(["index", "mjd"], kind="stable")
        tindex = lc["index"].to_numpy()
        uindex, starts = np.unique(tindex, return_index=True)
        stops = np.append(starts[1:], len(tindex))
        bands = lc["band"].to_numpy(dtype=object)
        mjd = lc["mjd"].to_numpy()
        zp = lc["zp"].to_numpy()
        flux = np.full(len(lc), np.nan)
        keep = np.ones(len(lc), dtype=bool)

        # --- 5. bandpass wavelength table
        if discard_bands:
            ubands, band_id = np.unique(bands, return_inverse=True)
            bps = [sncosmo.get_bandpass(b) for b in ubands]
            bminw = np.array([b.minwave() for b in bps])
            bmaxw = np.array([b.maxwave() for b in bps])

        # --- 4. parameters extracted once, one model
        if single_model:
            model = targets.template.get()
            cols = list(targets.get_template_columns())
            params = targets.data.loc[uindex, cols].to_numpy()
            magabs = targets.data.loc[uindex, "magabs"].to_numpy()
            pband, pmagsys, cosmo = targets.peak_absmag_band, targets.magsys, targets.cosmology

        # --- the only loop: sncosmo work
        niters = len(uindex)
        for i, (idx, start, stop) in tqdm( enumerate(zip(uindex, starts, stops)), total=niters) if progress_bar else enumerate(zip(uindex, starts, stops)):

            if single_model:
                model.set(**dict(zip(cols, params[i])))
                model.set_source_peakabsmag(absmag=magabs[i], band=pband, magsys=pmagsys, cosmo=cosmo)
            else:
                model = targets.get_target_template(index=idx, as_model=True, set_magabs=True)

            if discard_bands:
                bid = band_id[start:stop]
                ok = (bminw[bid] >= model.minwave()) & (bmaxw[bid] <= model.maxwave())
                keep[start:stop] = ok
                sel = start + np.flatnonzero(ok)
            else:
                sel = np.arange(start, stop)

            if len(sel) > 0:
                flux[sel] = model.bandflux(bands[sel], mjd[sel], zp=zp[sel], zpsys="ab")

        # --- 6. build the output once
        out = lc.loc[keep, ["index", "index_obs", "mjd", "band", "skynoise", "gain", "zp"] + field_names].copy()
        out["flux"] = flux[keep]
        out["fluxerr"] = np.sqrt(out["skynoise"]**2 + np.abs(out["flux"]) / out["gain"])
        out = out.set_index(["index", "index_obs"])
        if incl_error:
            rng = np.random.default_rng(seed)
            out["flux"] += rng.normal(loc=0, scale=out["fluxerr"])

        return out






# =================== #
    @classmethod
    def read_parquet(cls, parquetfile, survey=None, targets=None, **kwargs):
        """Loads a stored dataset.

        Only the observation data can be loaded this way,
        not the survey nor the targets (truth).

        Parameters
        ----------
        parquetfile: str
            path to the parquet file containing the dataset (pandas.DataFrame)

        survey: ``skysurvey.Survey`` (or child of), None
            survey that have been used to generate the dataset (if you know it)

        targets: ``skysurvey.Target`` (of child of), None
            target data corresponding to the true target parameters
            (as given by nature)

        **kwargs goes to `pandas.read_parquet`

        Returns
        -------
        class instance
            with a dataset loaded but maybe no survey nor targets

        See also
        --------
        :func:`from_targets_and_survey`: loads a dataset (observed data) given targets and survey
        """
        data = pandas.read_parquet(parquetfile, **kwargs)
        return cls(data, survey=survey, targets=targets)

    @classmethod
    def read_from_directory(cls, dirname, **kwargs):
        """Loads a directory containing the dataset, the survey and the targets.

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
        :func:`from_targets_and_survey`: loads a dataset (observed data) given targets and survey
        :func:`read_parquet`: loads a stored dataset
        """
        raise NotImplementedError("read_from_directory is not yet available.")

    # ============== #
    #   Method       #
    # ============== #
    # -------- #
    #  SETTER  #
    # -------- #
    def set_data(self, data):
        """Lightcurve data as observed by the survey.

        = It is unlikely you need to use that directly. =

        Parameters
        ----------
        data: `pandas.DataFrame`
            multi-index dataframe ((id, observation index))
            corresponding the concat of all targets observations

        Returns
        -------
        None

        See also
        --------
        :func:`read_parquet`: loads a stored dataset
        """
        self._data = data
        self._obs_index = None

    def set_targets(self, targets):
        """Set the targets.

        = It is unlikely you need to use that directly. =

        Parameters
        ----------
        targets: ``skysurvey.Target`` (of child of), None
            target data corresponding to the true target parameters
            (as given by nature)

        Returns
        -------
        None

        See also
        --------
        :func:`from_targets_and_survey`: loads a dataset (observed data) given targets and survey
        """
        self._targets = targets

    def set_survey(self, survey):
        """set the survey

        = It is unlikely you need to use that directly. =

        Parameters
        ----------
        survey: ``skysurvey.Survey`` (or child of), None
            survey that have been used to generate the dataset (if you know it)

        Returns
        -------
        None

        See also
        --------
        :func:`from_targets_and_survey`: loads a dataset (observed data) given targets and survey
        """
        self._survey = survey

    # -------- #
    #  GETTER  #
    # -------- #
    def get_data(self, add_phase=False, phase_range=None, index=None, redshift_key="z",
                detection=None, zp=None, join_bandday=False, join_how="first"):
        """ Tools to access the data with additional tools.

        Parameters
        ----------
        add_phase: bool
            should the phase information 'phase_obs' (obs-frame), 'phase' (rest-frame)
            be added to the dataframe assuming the input target's t0 and redshift ?

        phase_range: array
            min and max phases to be returned. Applied on phase (rest-frame).
            Setting this sets add_phase to True.

        index: `pandas.Index`, list, None
            select the index (targets id) you want.

        redshift_key: string
            name of the redshift column in the dset.targets.data.
             = ignored if add_phase is False =

        detection: bool, None
            should this be limited to (non)detected points only ?
            This follow the bool/None format:

            - detection=None: no selection
            - detection=False: only non-detected points
            - detection=True: only detected points

        zp: float
            get the simulated data in the given zp system

        join_bandday: bool
            if there are multiple observations per band and day (int of mjd) for a given target,
            should these be joined ? (see join_how).

        join_how: str
            join_bandday is True, how multiple observation should be considered ? (e.g., first).

        Returns
        -------
        `pandas.DataFrame`
        """
        if phase_range is not None:
            add_phase = True

        if index is not None:
            data = self.data.loc[index].copy()
        else:
            data = self.data.copy()
            index = data.index.levels[0]

        if join_bandday:
            index_colnames = data.index.names
            data["mjd_date"] = data["mjd"].astype("int")
            # make sure variance column exists as the variance add/mean/sum etc.
            if "fluxvar" not in data.columns:
                data["fluxvar"] = data["fluxerr"]**2
                fluxvar_to_be_removed = True
            else:
                fluxvar_to_be_removed = False

            gb_data = data.reset_index().groupby(by=["index", "band", "mjd_date"])
            if hasattr(gb_data, join_how):
                data = getattr(gb_data, join_how)().reset_index().set_index(index_colnames)
            else:
                raise NotImplementedError(f"gb_data.{join_how=} not implemented.")

            if join_how in ["mean", "sum"]:
                # overwrite fluxerr to respect the statistics
                data["fluxerr"] = np.sqrt(data["fluxvar"])

            if fluxvar_to_be_removed:
                _ = data.pop("fluxvar")

        if add_phase:
            target_info = self.targets.data.loc[index][["t0", redshift_key]]
            #        target_info.index = self._data_index # for merging
            data["phase_obs"] = data["mjd"] - target_info["t0"]
            data["phase"] = data["phase_obs"] / (1 + target_info[redshift_key])

        if phase_range is not None:
            data = data[data["phase"].between(*phase_range)]

        if detection is not None:
            flag_detection = (data["flux"] / data["fluxerr"]) >= 5
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

    def get_ndetection(self, phase_range=None, per_band=False, join_bandday=False, join_how="firt"):
        """get the number of detection for each lightcurves

        Basically computes the number of datapoints with (flux/fluxerr)>detlimit).

        Parameters
        ----------
        phase_range: array
            rest-frame phase range to be considered.

        per_band: bool
            should be computation be made per band ?
            if true it will then be per target *and* per band.

        join_bandday: bool
            if there are multiple observations per band and day (int of mjd) for a given target,
            should these be joined ? (see join_how).

        join_how: string
            specify how the bandday should be joined.
            = ignored if join_bandday is False =

        Returns
        -------
        `pandas.Series`
            the number of detected point per target (and per band if per_band=True)
        """

        data = self.get_data(phase_range=phase_range, detection=True,
                                 join_bandday=join_bandday,
                                 join_how=join_how)
        if per_band:
            groupby = [self._data_index, "band"]
        else:
            groupby = self._data_index

        ndetection = data.groupby(groupby).size()
        return ndetection

    def get_target_lightcurve(self, index, detection=None, phase_range=None):
        """Get the observation of the given target.

        = short cut to self.get_data(index=index) =

        Parameters
        ----------
        index : int, optional
            The index of the target whose light curve is to be taken. If None, a random index is chosen.

        detection: bool, None
            should this be limited to (non)detected points only ?
            This follow the bool/None format:

            - detection=None: no selection
            - detection=False: only non-detected points
            - detection=True: only detected points

        phase_range: array
            min and max phases to be returned. Applied on phase (rest-frame).
            Setting this sets add_phase to True.

        Returns
        -------
        `pandas.DataFrame`
            the lightcurve
        """
        return self.get_data(index=index, phase_range=phase_range, detection=detection)

    # -------- #
    #  PLOTTER #
    # -------- #
    def show_target_lightcurve(self, ax=None, fig=None, index=None, zp=25, lc_prop={}, bands=None, show_truth=True,
                               format_time=True, t0_format="mjd", phase_window=None, **kwargs):
        """Plot the light curve of a target.

        If `index` is None, a random index will be used. If `bands` is None,
        the target's observed band will be used.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`, optional
            The axes on which to plot the light curve. If None, a new figure and axes will be created.

        fig : `matplotlib.figure.Figure`, optional
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
        `matplotlib.figure.Figure`
            The figure object containing the light curve plot.
        """
        from matplotlib.colors import to_rgba

        from .config import get_band_color

        if format_time:
            from astropy.time import Time

        if index is None:
            rng = np.random.default_rng()
            index = rng.choice(self.obs_index)

        # Data
        obs_ = self.get_target_lightcurve(index).copy()
        if phase_window is not None:
            t0 = self.targets.data["t0"].loc[index]
            phase_window = np.asarray(phase_window) + t0
            obs_ = obs_[obs_["mjd"].astype("float").between(*phase_window)]

        coef = 10 ** (-(obs_["zp"] - zp) / 2.5)
        obs_["flux_zp"] = obs_["flux"] * coef
        obs_["fluxerr_zp"] = obs_["fluxerr"] * coef

        if len(obs_) == 0:
            warnings.warn(f"No detections for the SN index={index} (detections possibly outside phase_window).")
            return None

        # Model
        if bands is None:
            bands = np.unique(obs_["band"])

        # = axes and figure = #
        if ax is None:
            if fig is None:
                import matplotlib.pyplot as plt

                fig = plt.figure(figsize=[7, 4])
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure

        colors = get_band_color(bands)
        if show_truth:
            fig = self.targets.show_lightcurve(bands, ax=ax, fig=fig, index=index, format_time=format_time,
                                               t0_format=t0_format, zp=zp, colors=colors, zorder=2, **lc_prop)
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
            times = (
                obs_band["mjd"]
                if not format_time
                else Time(obs_band["mjd"], format=t0_format).datetime
            )
            ax.scatter(times, obs_band["flux_zp"], color=color_, zorder=4, **kwargs)
            ax.errorbar(times, obs_band["flux_zp"], yerr=obs_band["fluxerr_zp"], ls="None", marker="None",
                        ecolor=ecolor, zorder=3, **kwargs)

        return fig

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
        if not hasattr(self, "_obs_index") or self._obs_index is None:
            self._obs_index = self.data.index.get_level_values(0).unique().sort_values()

        return self._obs_index
