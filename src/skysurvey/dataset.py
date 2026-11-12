"""Data as observed: lightcurves of targets observed by a survey.

:class:`DataSet` joins information from targets (true parameters) and a
survey (what has been observed when) to generate realistic lightcurve
observations.
"""

import numpy as np
import pandas
import sncosmo
import warnings

from .target.collection import TargetCollection

__all__ = ["DataSet"]

# ================== #
#  Building blocks   #
# ================== #

# ================== #
#                    #
#    DataSet         #
#                    #
# ================== #
class DataSet(object):
    """Realistic transient lightcurves given true targets and survey logs.

    This class provides methods to load, manipulate, and visualize lightcurve
    data based on target and survey information.

    The classmethod :meth:`DataSet.from_targets_and_survey` should be favored
    for loading the dataset.

    Parameters
    ----------
    data : pandas.DataFrame
        Multi-index dataframe corresponding to the concatenation of all
        targets observations.

    targets : skysurvey.Target, optional
        Target data corresponding to the true target parameters (as given by
        nature). The default is None.

    survey : skysurvey.Survey, optional
        Survey that has been used to generate the dataset (if known).
        The default is None.

    See Also
    --------
    from_targets_and_survey : Load a dataset given targets and a survey.
    read_parquet : Load a stored dataset.
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
        """Load a dataset (observed data) given targets and a survey.

        This first matches the targets (given ``targets.data[['ra', 'dec']]``)
        with the survey to find which target has been observed with which
        field. The match uses all the field identifiers of the survey
        (``survey.fieldids.names``, e.g. a field and a CCD id): any number of
        them is supported. Then it simulates the targets lightcurves given the
        observing data (``survey.data``).

        Parameters
        ----------
        targets : skysurvey.Target, list, or skysurvey.TargetCollection
            Target data corresponding to the true target parameters (as given
            by nature). A list (or tuple) of targets is converted into a
            :class:`~skysurvey.target.collection.TargetCollection`.

        survey : skysurvey.Survey
            Sky observation (what was observed when with which situation).

        incl_error : bool, optional
            Include error in the lightcurve. If False, the flux is the true
            model flux. The default is True.

        phase_range : list or None, optional
            Rest-frame phase range to be used for simulating the lightcurves.
            If None, no cut is applied on time range for the logs.
            The default is [-50, +200].

        progress_bar : bool, optional
            Whether to display a progress bar (uses tqdm) associated to the
            generation of targets. The default is False.

        seed : None, int, or numpy.random.Generator, optional
            Seed passed to :func:`numpy.random.default_rng` to draw the flux
            noise; ignored if `incl_error` is False. If None, a fresh seed is
            pulled. The default is None.

        discard_bands : bool, optional
            If True, discard the bands that include wavelengths for which the
            (observer-frame) target SED is not defined. This prevents
            crashing the code due to an error from `sncosmo`.
            The default is True.

        Returns
        -------
        skysurvey.DataSet
            Instance of a `DataSet` loaded from the given targets.
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
        """Simulate the observed lightcurves of targets given a survey.

        Parameters
        ----------
        targets : skysurvey.Target
            Target data corresponding to the true target parameters (as given
            by nature).

        survey : skysurvey.Survey
            Sky observation (what was observed when with which situation).

        progress_bar : bool, optional
            Whether to display a progress bar (uses tqdm). The default is False.

        incl_error : bool, optional
            Include error in the lightcurve. If False, the flux is the true
            model flux. The default is True.

        phase_range : list or None, optional
            Rest-frame phase range to be used for simulating the lightcurves.
            If None, no cut is applied. The default is [-50, 200].

        seed : None, int, or numpy.random.Generator, optional
            Seed passed to :func:`numpy.random.default_rng` to draw the flux
            noise; ignored if `incl_error` is False. The default is None.

        discard_bands : bool, optional
            If True, discard the observations in bands that include
            wavelengths for which the (observer-frame) target SED is not
            defined. The default is True.

        single_model : bool, optional
            If True, a single template model is created and its parameters
            are updated for each target (faster). If False, a new model is
            built for each target. The default is True.

        Returns
        -------
        pandas.DataFrame
            Multi-index (``index``, ``index_obs``) dataframe of the simulated
            observations, including ``flux`` and ``fluxerr`` columns.
        """
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
            # magobs is part of the model dag: use it directly if available
            use_magobs = "magobs" in targets.data
            peakmag = targets.data.loc[uindex, "magobs" if use_magobs else "magabs"].to_numpy()
            pband, pmagsys, cosmo = targets.peak_absmag_band, targets.magsys, targets.cosmology

        # --- the only loop: sncosmo work
        niters = len(uindex)
        for i, (idx, start, stop) in tqdm( enumerate(zip(uindex, starts, stops)), total=niters) if progress_bar else enumerate(zip(uindex, starts, stops)):

            if single_model:
                model.set(**dict(zip(cols, params[i])))
                if use_magobs:
                    model.set_source_peakmag(m=peakmag[i], band=pband, magsys=pmagsys)
                else:
                    model.set_source_peakabsmag(absmag=peakmag[i], band=pband, magsys=pmagsys, cosmo=cosmo)
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


    @classmethod
    def read_parquet(cls, parquetfile, survey=None, targets=None, **kwargs):
        """Load a stored dataset.

        Only the observation data can be loaded this way, not the survey nor
        the targets (truth), which can be provided separately.

        Parameters
        ----------
        parquetfile : str
            Path to the parquet file containing the dataset
            (pandas.DataFrame).

        survey : skysurvey.Survey, optional
            Survey that has been used to generate the dataset (if known).
            The default is None.

        targets : skysurvey.Target, optional
            Target data corresponding to the true target parameters (as given
            by nature). The default is None.

        **kwargs
            Passed to :func:`pandas.read_parquet`.

        Returns
        -------
        skysurvey.DataSet
            Instance with the dataset loaded, but possibly no survey nor
            targets.

        See Also
        --------
        from_targets_and_survey : Load a dataset given targets and a survey.
        """
        data = pandas.read_parquet(parquetfile, **kwargs)
        return cls(data, survey=survey, targets=targets)

    @classmethod
    def read_from_directory(cls, dirname, **kwargs):
        """Load a directory containing the dataset, the survey and the targets.

        Not implemented yet.

        Parameters
        ----------
        dirname : str
            Path to the directory.

        **kwargs
            Currently unused.

        Returns
        -------
        skysurvey.DataSet
            Instance of the loaded dataset.

        Raises
        ------
        NotImplementedError
            Always, as this is not implemented yet.

        See Also
        --------
        from_targets_and_survey : Load a dataset given targets and a survey.
        read_parquet : Load a stored dataset.
        """
        raise NotImplementedError("read_from_directory is not yet available.")

    # ============== #
    #   Method       #
    # ============== #
    # -------- #
    #  SETTER  #
    # -------- #
    def set_data(self, data):
        """Set the lightcurve data as observed by the survey.

        It is unlikely you need to use this directly.

        Parameters
        ----------
        data : pandas.DataFrame
            Multi-index dataframe (target id, observation index)
            corresponding to the concatenation of all targets observations.

        See Also
        --------
        read_parquet : Load a stored dataset.
        """
        self._data = data
        self._obs_index = None


    def set_targets(self, targets):
        """Set the targets.

        It is unlikely you need to use this directly.

        Parameters
        ----------
        targets : skysurvey.Target or None
            Target data corresponding to the true target parameters (as given
            by nature).

        See Also
        --------
        from_targets_and_survey : Load a dataset given targets and a survey.
        """
        self._targets = targets


    def set_survey(self, survey):
        """Set the survey.

        It is unlikely you need to use this directly.

        Parameters
        ----------
        survey : skysurvey.Survey or None
            Survey that has been used to generate the dataset (if known).

        See Also
        --------
        from_targets_and_survey : Load a dataset given targets and a survey.
        """
        self._survey = survey


    # -------- #
    #  GETTER  #
    # -------- #
    def get_data(self, add_phase=False, phase_range=None,
                index=None, redshift_key="z",
                detection=None, zp=None,
                join_bandday=False, join_how="first", boundary=None):
        """Get the observation data with optional selections and additions.

        Parameters
        ----------
        add_phase : bool, optional
            Whether the phase information ``phase_obs`` (observer-frame) and
            ``phase`` (rest-frame) should be added to the dataframe, assuming
            the input target's t0 and redshift. The default is False.

        phase_range : array_like, optional
            Min and max phases to be returned. Applied on phase (rest-frame).
            Setting this sets `add_phase` to True. The default is None.

        index : pandas.Index, list, or None, optional
            Index (target ids) to select. If None, all targets are returned.
            The default is None.

        redshift_key : str, optional
            Name of the redshift column in ``self.targets.data``; ignored if
            `add_phase` is False. The default is "z".

        detection : bool or None, optional
            Whether to limit to (non-)detected points only
            (detection means flux/fluxerr >= 5, after `join_bandday`):

            - None: no selection
            - False: only non-detected points
            - True: only detected points

            The default is None.

        zp : float, optional
            If given, convert the flux and fluxerr to this zero point system.
            The default is None.

        join_bandday : bool or int, optional
            Join the observations of each target taken in the same band and the
            same night (True) or the same `join_bandday` nights (int, e.g. 3; see
            `join_how` and `boundary`). The data then have one row per (target,
            band, night), with the ``night`` (first night of the window) and the
            number of joined observations ``nvisits``. With N nights, the windows
            are consecutive N-night blocks starting at the first night of the
            survey logs (or of ``self.data`` without survey): the same for all
            targets. Co-adding over several nights averages a varying flux: keep N
            short with respect to the transient timescale (N observer-frame days
            are N/(1+z) rest-frame days). False (or 0) means no join. The default
            is False.

        join_how : str, optional
            How the observations of a band and night(s) are joined, if
            `join_bandday`:

            - "first", "last": the first (last) observation.
            - "coadd" (or "add", "sum"): co-addition, i.e. the inverse-variance
              weighted mean flux (and mjd) with error ``1/sqrt(sum(1/fluxerr**2))``
              (see ``DataSet._coadd_observations_``). The fluxes are first converted
              to a common zero point (`zp`, or that of the first observation).

            The default is "first".

        boundary : float, optional
            Time of day separating two nights, as a fraction of the UT day,
            used with `join_bandday`. If None, from the survey logs
            (``self.survey.get_night_boundary()``), or 0 (0h UT) if there is no
            survey. The default is None.

        Returns
        -------
        pandas.DataFrame
            The (selected) observation data.

        Raises
        ------
        ValueError
            If `join_how` is not one of the above, or if `join_bandday` is
            neither a bool nor a positive integer.

        Warns
        -----
        UserWarning
            If `join_bandday` without `boundary` nor survey (nights change at
            0h UT), or if observations of different zero points are co-added
            without `zp` (converted to the zero point of the first one).
        """
        if phase_range is not None:
            add_phase = True

        # a single target: selected as a list to keep the (target, observation) index
        # levels while processing (e.g. join_bandday); the target level is dropped at the end.
        single_target = index is not None and np.ndim(index) == 0
        if index is not None:
            data = self.data.loc[np.atleast_1d(index)].copy()
        else:
            data = self.data.copy()

        if zp is not None:
            coef = 10 ** (-(data["zp"].values - zp) / 2.5)
            data["flux"] *= coef
            data["fluxerr"] *= coef
            data["zp"] = zp

        if join_bandday:
            data = self._join_bandday_(data, join_how=join_how, boundary=boundary,
                                       nnights=join_bandday)

        if add_phase:
            # t0 and redshift of the target of each observation (by target id, whatever the index names)
            target_info = self.targets.data.loc[data.index.get_level_values(0), ["t0", redshift_key]]
            data["phase_obs"] = data["mjd"].to_numpy() - target_info["t0"].to_numpy()
            data["phase"] = data["phase_obs"] / (1 + target_info[redshift_key].to_numpy())

        if phase_range is not None:
            data = data[data["phase"].between(*phase_range)]

        if detection is not None:
            flag_detection = (data["flux"] / data["fluxerr"]) >= 5
            if detection:
                data = data[flag_detection]
            else:
                data = data[~flag_detection]

        if single_target: # as self.data.loc[index]: indexed by observation only
            data = data.droplevel(0)

        return data

    @staticmethod
    def _coadd_observations_(data, by, first_columns=None):
        """Co-add the observations sharing the same `by` values.

        The co-added flux is the inverse-variance weighted mean of the fluxes,
        ``sum(w_i flux_i) / sum(w_i)`` with ``w_i = 1/fluxerr_i**2``, and its error
        ``1/sqrt(sum(w_i))``: N observations of equal errors at S/N each give
        S/N x sqrt(N). The fluxes must share the same zero point.

        Parameters
        ----------
        data : pandas.DataFrame
            Observations with (at least) the `by` columns, ``mjd``, ``flux`` and
            ``fluxerr``.

        by : list of str
            Columns defining the observations to co-add together, e.g.
            ``["index", "band", "night"]``.

        first_columns : list of str, optional
            Other columns to keep, with the value of the first observation of each
            group (e.g. ``["zp"]``). The default is None.

        Returns
        -------
        pandas.DataFrame
            One row per group (columns `by`): weighted mean ``mjd`` and ``flux``,
            ``fluxerr``, the number of co-added observations ``nvisits``, and the
            `first_columns`.
        """
        data = data.assign(weight=data["fluxerr"]**-2) # inverse variance
        data = data.assign(wflux=data["flux"] * data["weight"], wmjd=data["mjd"] * data["weight"])

        grouped = data.groupby(by)
        sums = grouped[["weight", "wflux", "wmjd"]].sum()
        coadded = pandas.DataFrame({"mjd": sums["wmjd"] / sums["weight"],
                                    "flux": sums["wflux"] / sums["weight"],
                                    "fluxerr": sums["weight"]**-0.5,
                                    "nvisits": grouped.size()})
        if first_columns is not None:
            coadded = coadded.join(grouped[list(first_columns)].first())

        return coadded.reset_index()


    def _join_bandday_(self, data, join_how="first", zp=None, boundary=None, nnights=1):
        """Join the observations per target, band and `nnights` nights (see :meth:`get_data`)."""
        # True is 1 night; otherwise a whole number of nights.
        if isinstance(nnights, (bool, np.bool_)):
            nnights = int(nnights)
        if not isinstance(nnights, (int, np.integer)) or nnights < 1:
            raise ValueError(f"join_bandday must be a bool or a positive integer (number of nights), not {nnights!r}")

        if boundary is None:
            if self.survey is not None and hasattr(self.survey, "get_night_boundary"):
                boundary = self.survey.get_night_boundary()
            else:
                warnings.warn("join_bandday without survey nor boundary: nights change at 0h UT.")
                boundary = 0

        index_names = list(data.index.names) # (target, observation)
        data = data.reset_index()
        data["night"] = np.floor(data["mjd"] - boundary).astype(int)
        if nnights > 1: # N-night windows, from the first night of the survey (the same for all targets)
            survey_data = getattr(self.survey, "data", None)
            if isinstance(survey_data, pandas.DataFrame) and "mjd" in survey_data:
                first_mjd = survey_data["mjd"].min()
            else:
                first_mjd = self.data["mjd"].min()

            first_night = int(np.floor(first_mjd - boundary))
            data["night"] = first_night + (data["night"] - first_night) // nnights * nnights

        by = [index_names[0], "band", "night"]

        if join_how in ["first", "last"]:
            grouped = data.groupby(by)
            joined = getattr(grouped, join_how)()
            joined["nvisits"] = grouped.size()
            joined = joined.reset_index()

        elif join_how in ["add", "coadd", "sum"]:
            # co-addition requires a common zero point (none: fluxes assumed homogeneous).
            if "zp" in data:
                if zp is None:
                    zp = data["zp"].iloc[0]
                    if data["zp"].nunique() > 1:
                        warnings.warn(f"observations of different zero points are co-added at zp={zp}: set zp to choose it.")

                coef = 10 ** (-(data["zp"] - zp) / 2.5)
                data["flux"], data["fluxerr"], data["zp"] = data["flux"] * coef, data["fluxerr"] * coef, zp

            first_columns = [index_names[1]] + [k for k in ["zp", "zpsys"] if k in data]
            joined = self._coadd_observations_(data=data, by=by, first_columns=first_columns)

        else:
            raise ValueError(f"join_how must be 'first', 'last' or 'coadd' (or 'add', 'sum'), not {join_how=}")

        return joined.set_index(index_names)

    def get_ndetection(self, phase_range=None, per_band=False,
                        join_bandday=False, join_how="coadd",
                        boundary=None):
        """Get the number of detections for each lightcurve.

        Computes the number of datapoints with (flux/fluxerr) >= 5
        (see :meth:`get_data`). With ``join_bandday=True, join_how="coadd"``,
        the observations of each band and night are co-added first: e.g. 10
        observations at 4 sigma in one night make a 12.6 sigma detection.

        Parameters
        ----------
        phase_range : array_like, optional
            Rest-frame phase range to be considered. The default is None.

        per_band : bool, optional
            Whether the computation should be made per band. If True, it is
            made per target *and* per band. The default is False.

        join_bandday : bool or int, optional
            Join the observations of each target taken in the same band and the
            same night (True) or the same N nights (int) before counting (see
            :meth:`get_data`). The default is False.

        join_how : str, optional
            How the observations of a band and night are joined: "first",
            "last" or "coadd" (see :meth:`get_data`); ignored if `join_bandday`
            is False. The default is "coadd".

        boundary : float, optional
            Time of day separating two nights (see :meth:`get_data`).
            The default is None.

        Returns
        -------
        pandas.Series
            The number of detected points per target (and per band if
            `per_band` is True).
        """
        data = self.get_data(phase_range=phase_range, detection=True,
                             join_bandday=join_bandday, join_how=join_how,
                             boundary=boundary)
        if per_band:
            groupby = [self._data_index, "band"]
        else:
            groupby = self._data_index

        ndetection = data.groupby(groupby).size()
        return ndetection

    def get_target_lightcurve(self, index, detection=None, phase_range=None,
                                coadd=False, **kwargs):
        """Get the observations of the given target.

        Shortcut to ``self.get_data(index=index)``.

        Parameters
        ----------
        index : int
            The index of the target whose lightcurve is to be taken.

        detection : bool or None, optional
            Whether to limit to (non-)detected points only:

            - None: no selection
            - False: only non-detected points
            - True: only detected points

            The default is None.

        phase_range : array_like, optional
            Min and max phases to be returned. Applied on phase (rest-frame).
            The default is None.

        coadd : bool or int, optional
            Co-add the observations per band and night (True) or per band and N
            nights (int), i.e. ``join_bandday=coadd, join_how="coadd"`` (see
            :meth:`get_data`). The default is False.

        **kwargs
            Passed to :meth:`get_data` (e.g. `boundary`, `zp`).

        Returns
        -------
        pandas.DataFrame
            The lightcurve.

        See Also
        --------
        get_data : Get the observation data.
        """
        if coadd:
            kwargs["join_bandday"] = coadd
            kwargs["join_how"] = "coadd"

        return self.get_data(index=index, phase_range=phase_range, detection=detection, **kwargs)

    # -------- #
    #  PLOTTER #
    # -------- #
    def show_target_lightcurve(self, ax=None, fig=None, index=None, zp=25, lc_prop={},
                                bands=None, show_truth=True, coadd=False,
                               format_time=True, t0_format="mjd", phase_window=None, **kwargs):
        """Plot the lightcurve of a target.

        If `index` is None, a random index is used. If `bands` is None, the
        target's observed bands are used.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot the lightcurve. If None, a new figure
            and axes are created. The default is None.

        fig : matplotlib.figure.Figure, optional
            The figure on which to plot the lightcurve (ignored if `ax` is
            given). If None, a new figure is created. The default is None.

        index : int, optional
            The index of the target whose lightcurve is to be plotted.
            If None, a random observed index is chosen. The default is None.

        zp : float, optional
            Zero point for the flux conversion. The default is 25.

        lc_prop : dict, optional
            Additional properties passed to the true lightcurve plotting
            function (``self.targets.show_lightcurve``). The default is {}.

        bands : list of str, optional
            The bands to plot. If None, all observed bands for the target are
            used. The default is None.

        show_truth : bool, optional
            Whether to show the true lightcurve. The default is True.

        coadd : bool or int, optional
            Show the observations co-added per band and night (True) or per band
            and N nights (int) (see :meth:`get_target_lightcurve`). The default
            is False.

        format_time : bool, optional
            Whether to format the time axis as dates. The default is True.

        t0_format : str, optional
            The format of the reference time. The default is "mjd".

        phase_window : array_like, optional
            The (observer-frame) phase window, relative to t0, to plot.
            If None, the entire lightcurve is plotted. The default is None.

        **kwargs
            Passed to the scatter and errorbar plotting functions.

        Returns
        -------
        matplotlib.figure.Figure or None
            The figure containing the lightcurve plot, or None if there are
            no data points to show.
        """
        from matplotlib.colors import to_rgba

        from .config import get_band_color

        if format_time:
            from astropy.time import Time

        if index is None:
            rng = np.random.default_rng()
            index = rng.choice(self.obs_index)

        # Data
        obs_ = self.get_target_lightcurve(index, coadd=coadd).copy()
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
        """Index of the observed targets."""
        if not hasattr(self, "_obs_index") or self._obs_index is None:
            self._obs_index = self.data.index.get_level_values(0).unique().sort_values()

        return self._obs_index
