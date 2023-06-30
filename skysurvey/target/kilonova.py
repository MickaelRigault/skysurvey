from .core import Transient


__all__ = ["Kilonova"]


class Kilonova( Transient ):

    _KIND = "kilonova"
    _TEMPLATE_SOURCE = "salt2"
    _VOLUME_RATE = 1
    
