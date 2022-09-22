""" Core Sampling library """

import numpy as np
import inspect

class Sampling( object ):
    """ 
    Create your own Sampling
    ------------------------
    
    # You need to create a sampling class for a new parameters
    # named {name}. Do
    
    class Sampling_{name}( Sampling ):
        _MIN = float  # optional
        _MAX = float  # optional
        _STEP = float # optional
        
    # Now you can create ways to sample your parameters
    #  You have two ways:
    #  1. from a pdf function, then do
    def getpdf_{name_of_model}(xx=None, *args, **kwargs):
        ''' '''
        pdf = ...
        return xx, pdf
        
    # 2. directly from a sampling method
    def {name_of_model}(size=None, *args, **kwargs):
        ''' '''
        return array (the sampling)
    
    # -> if 2. exists, sampling comes from there.
    
    """
    
    _MIN = 0
    _MAX = 1
    _STEP = 1e-3

    @classmethod
    def get_default_xx(cls):
        """ """
        return np.arange(cls._MIN, cls._MAX, cls._STEP)
        
    @classmethod
    def draw(cls, model, size=None, xx=None, **kwargs):
        """ """
        if hasattr(cls,f"{model}"):
            func = getattr(cls, model)
            if "size" in list(inspect.getfullargspec(func).args):
                kwargs["size"] = size
            return func(**kwargs)
        
        elif hasattr(cls,f"getpdf_{model}"):
            return cls._draw_from_pdf(model, size=size, xx=xx, **kwargs)
        else:
            raise NotImplementedError(f"No model or getpdf_model for {model}")
    
    @classmethod
    def _draw_from_pdf(cls, model, size=None, **kwargs):
        """ """
        xx, pdf = getattr(cls, f"getpdf_{model}")(**kwargs)
        return np.random.choice(xx, size=size, p=pdf/pdf.sum())
    
    @classmethod
    def get_pdf(cls, model, **kwargs):
        """ """
        if hasattr(cls,f"getpdf_{model}"):
            return getattr(cls,f"getpdf_{model}")(**kwargs)
        # Error
        elif hasattr(cls,f"{model}"):
            # Sampler yes, but no pdf
            raise NotImplementedError(f"No pdf available for {model}.")
        else:
            raise NotImplementedError(f"No model named {model}")
