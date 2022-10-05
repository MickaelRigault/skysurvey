
import numpy as np
import pandas

from .core import Target, Transient
from .timeserie import TSTransient

def reshape_values(values, shape):
    """ """
    values = np.atleast_1d(values)
    if len(values) == 1:
        values = np.resize(values, shape)
    assert len(values) == shape
    # success
    return values


class TargetCollection( object ):
    _COLLECTION_OF = target.Target
    
    def __init__(self, targets):
        """ """
        self.set_targets(targets)

    # ============= #
    #  Collection   #
    # ============= #            
    def call_down(self, which, margs=None, **kwargs):
        """ """
        if margs is not None:
            margs = reshape_values(margs, self.ntargets)
            return [getattr(t, which)(marg_, **kwargs) for marg_, t in zip(margs, self.targets)]
            
        return [attr if not callable(attr:=getattr(t, which)) else\
                attr(**kwargs) 
                for t in self.targets]
    
    # ============= #
    #  Methods      #
    # ============= #    
    def set_targets(self, targets):
        """ """
        self._targets = np.atleast_1d(targets)
            
    def get_model_parameters(self, entry, key, default=None):
        """ """
        return self.call_down("get_model_parameter", 
                              entry=entry, key=key, default=default)
    # ============= #
    #  Properties   #
    # ============= #
    @property
    def targets(self):
        """ list of transients """
        return self._targets
    
    @property
    def ntargets(self):
        """ number of targets """
        return len(self.targets)
    
    @property
    def target_ids(self):
        """ targets id """
        return np.arange(self.ntargets)
    
    @property
    def models(self):
        """ list of the target models """
        return self.call_down("model")
    
class TransientCollection( TargetCollection ):
    _COLLECTION_OF = target.Target
    
    # ============= #
    #  Methods      #
    # ============= #
    def get_rates(self, z, relative=False, **kwargs):
        """ """
        rates = self.call_down("get_rate", margs=z, **kwargs)
        if relative:
            rates /= np.nansum(rates)
        return rates
    
    def draw(self, size=None, zmax=None,
             tstart=None, tstop=None, nyears=None,
             inplace=True, **kwargs):
        """ 
        """
        if size is not None:
            size_ = np.asarray(size*self.get_rates(0.1, relative=True), dtype="int")
            if np.sum(size_) < size:
                size_[-1] += size-np.sum(size_)
            size = size_
                
            # solving int issue
            
        draws = self.call_down("draw", margs=size, 
                              tstart=tstart, tstop=tstop, nyears=nyears, inplace=False, 
                              **kwargs)
        
        data = pandas.concat(draws, keys=self.target_ids, axis=0)
        if inplace:
            self._data = data
            
        return data
    # ============= #
    #  Properties   #
    # ============= #
    @property
    def rates(self):
        """ list of transients """
        return self.call_down("rate")
    
    @property
    def data(self):
        """ """
        if not hasattr(self,"_data"):
            return None
        return self._data
    
class TSTransientCollection( TransientCollection ):
    _COLLECTION_OF = target.TSTransient
        
    @classmethod
    def from_draw(cls, sources, size=None, nyears=None, 
                  rates=1e3, magabs=None, magscatter=None, **kwargs):
        """ """
        this = cls.from_sncosmo_sources(sources, rates=rates,
                                        magabs=magabs, 
                                        magscatter=magscatter)
        _ = this.draw(size=size, nyears=nyears, inplace=True,
                      **kwargs)
        return this
        
    @classmethod
    def from_sncosmo_sources(cls, sources, rates=1e3, 
                             magabs=None, magscatter=None):
        """ loads the instance from a list of sources
        (and relative rates)
        """
        # make sure the sizes match
        rates = reshape_values(rates, len(sources))
        transients = [cls._COLLECTION_OF.from_sncosmo_source(source_, rate_)
                     for source_, rate_ in zip(sources, rates)]
        
        # Change the model.
        if magabs is not None:
            magabs = reshape_values(magabs, len(sources))
            print(magabs)
            _ = [t.change_model_parameter(magabs={"loc":magabs_}) 
                 for t, magabs_ in zip(transients, magabs)]
            
        if magscatter is not None:
            magscatter = reshape_values(magscatter, len(sources))
            _ = [t.change_model_parameter(magabs={"scale":magscatter_}) 
                 for t, magscatter_ in zip(transients, magscatter)]
            
        # and loads it
        return cls(transients)
