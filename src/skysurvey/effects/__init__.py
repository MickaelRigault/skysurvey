"""Propagation effects (dust, scatter) applied to target templates.

Pre-built effects are available as ``mw_extinction`` (Milky Way dust) and
``host_dust`` (host galaxy dust).
"""
from .core import Effect
mw_extinction = Effect.from_name("mw")
host_dust = Effect.from_name("hostdust")
