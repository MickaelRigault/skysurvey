import sncosmo
from . import milkyway

# ============== #
#  Effect Class  #
# ============== #
class Effect( object ):
    
    def __init__(self, 
                 effect=None, 
                 name=None,
                 frame=None,
                 model={}):
        """ """
        self._effect = effect
        self._name = name
        self._frame = frame
        self._model = model
    
    @classmethod
    def from_sncosmo(cls, effect, name, frame, model={}):
        """ 
        effect : `~sncosmo.PropagationEffect`
            Propagation effect.
            
        name : str
            Name of the effect.
            
        frame : str
            'rest': rest-frame 
            'obs': observator-frame
            'free':
            
        """
        return cls(effect, name, frame, model=model)
    
    @classmethod
    def from_name(cls, name, which=None):
        """ shortcut to usual effects """

        
        name = name.lower().replace("_","") # case insensitive, no "_"
        
        # Milky Way dust map
        if name in ["mw","mwebv","mwdust", "mwmap"]:
            if which is None or which == "ccm89":
                effect = sncosmo.CCM89Dust()
            else: 
                raise NotImplementedError("only ccm89 dust law implemented")
                
            name = "mw"
            frame = "obs"
            model = milkyway.mwebv_model
            
        # Host dust
        elif name in ["hostdust","dust"]:
            if which is None or which.lower() == "ccm89":
                effect = sncosmo.CCM89Dust()
            else: 
                raise NotImplementedError("only ccm89 dust law implemented")
                
            effect = sncosmo.CCM89Dust()
            name = "hostdust"
            frame = "rest"
            model = {}
            
        # SNIa color scatter
        
        elif name.lower() in ["scatter", "scattercolor", "color_scatter", "colorscatter"]:
            name = "colorscatter"
            frame = "rest"
            model = {}
            if which.lower() == "g10":
                from . import scatter
                effect = scatter.ColorScatter_G10.from_saltsource()
            elif which.lower() == "c11":
                from . import scatter
                effect = scatter.ColorScatter_C11()
            else:
                raise ValueError(f"unknown colorscatter {which}")
            
        else:
            raise NotImplementedError(f"cannot parse the input effect: {name}")
            
        return cls.from_sncosmo(effect, name, frame, model=model)
    
    # ============= #
    #  Internal     #
    # ============= #    
    def __repr__(self):
        """ """        
        return self.__str__()
    
    def __str__(self):
        """ """
        import pprint
        out = { "effect": self.effect,
                "name":self.name,
                "frame":self.frame,
                "model": self.model
               }
        return pprint.pformat(out, sort_dicts=False)

    # ============= #
    #  Properties   #
    # ============= #
    @property
    def effect(self):
        return self._effect
    
    @property
    def name(self):
        return self._name
    
    @property
    def frame(self):
        """ """
        return self._frame

    @property
    def model(self):
        """ """
        return self._model
