"""Fast batched SALT2 fit of a DataSet with saltjax (JAX)."""


def fit_salt_jax(dataset, indexes=None, modelcov=True, phase_range=[-10, 40], guess="data",
                 **kwargs):
    """Fit SALT2 on the lightcurves of a dataset, all targets at once, with JAX.

    This is a fast equivalent of :func:`skysurvey.lcfit.fit_salt` (that calls
    ``sncosmo.fit_lc`` target by target), using the `saltjax
    <https://github.com/MickaelRigault/saltjax>`_ package (optional
    dependency): same model, same chi2, same model-covariance procedure.
    Parameters are not bounded.

    Parameters
    ----------
    dataset : skysurvey.DataSet
        Dataset containing SALT2-based targets and their lightcurves.

    indexes : list or None, optional
        Targets to fit. If None, all observed targets (``dataset.obs_index``).
        The default is None.

    modelcov : bool, optional
        Include the SALT2 model covariance in the chi2, as in
        ``sncosmo.fit_lc(..., modelcov=True)``. The default is True.

    phase_range : list or None, optional
        Rest-frame phase range (relative to the true t0) of the data used.
        The default is [-10, 40].

    guess : {'data', 'truth'}, optional
        Initial guess: from the data (default) or the true t0, x1 and c.
        See :func:`saltjax.fit_salt`. The default is 'data'.

    **kwargs
        Passed to :func:`saltjax.fit_salt` (e.g. ``progress_bar``,
        ``batch_points``, ``c_max``, ``dz``, ``minsnr``, ``nrefit``).

    Returns
    -------
    pandas.DataFrame
        One row per fitted target, with the same columns as
        :func:`skysurvey.lcfit.fit_salt` (``z``, ``t0``, ``x0``, ``x1``, ``c``,
        their ``_err``, ``cov_{p}{q}`` and the fixed ``mwebv`` and ``mwr_v``),
        plus ``chi2``, ``ndof``, ``converged`` and ``nrefit``.

    Raises
    ------
    ImportError
        If saltjax is not installed.

    NotImplementedError
        If the targets use another source than SALT2, or effects other than
        the Milky Way ``CCM89Dust`` in the observer frame.

    See Also
    --------
    skysurvey.lcfit.fit_salt : The sncosmo-based (target by target) equivalent.
    """
    try:
        import saltjax
    except ImportError as e:
        raise ImportError("fit_salt_jax requires saltjax: "
                          "pip install saltjax") from e

    if indexes is None:
        indexes = dataset.obs_index

    indexes = list(indexes)
    template = dataset.targets.get_target_template(indexes[0]).sncosmo_model

    # Milky Way dust (the only supported effect)
    mwebv_key, mw_r_v = None, 3.1
    for effect, name, frame in zip(template.effects, template.effect_names,
                                   template._effect_frames):
        if type(effect).__name__ == "CCM89Dust" and frame == "obs":
            mwebv_key, mw_r_v = f"{name}ebv", float(template.get(f"{name}r_v"))
        else:
            raise NotImplementedError(f"effect {type(effect).__name__} ({frame}) not supported")

    return saltjax.fit_salt(dataset.data, dataset.targets.data, indexes=indexes,
                            modelcov=modelcov, phase_range=phase_range, guess=guess,
                            source=template.source, mwebv_key=mwebv_key, mw_r_v=mw_r_v,
                            **kwargs)
