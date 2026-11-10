"""Simulate astronomical transients and the surveys that observe them.

skysurvey combines three objects:

- a `Target` (e.g. `SNeIa`), drawing the true parameters of the transients,
- a `Survey` (e.g. `GridSurvey`), holding what has been observed when,
- a `DataSet`, joining both to build realistic lightcurves.
"""
__version__ = "1.4.2"

import os
_PACKAGE_PATH = os.path.dirname( os.path.realpath(__file__) )

from .dataset import * # noqa: F403, E402
from .survey import * # noqa: F403, E402
from .target import * # noqa: F403, E402
from .effects import Effect # noqa: F401, E402
# info
