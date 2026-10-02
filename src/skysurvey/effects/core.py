"""Effect class to build sncosmo effects with associated parameter models."""

import sncosmo
from . import milkyway
from . import hostdust

# ============== #
#  Effect Class  #
# ============== #
class Effect( object ):
    """Effect (e.g. dust, scatter) to be applied to a transient template.

    Parameters
    ----------
    effect : sncosmo.PropagationEffect, optional
        Propagation effect. The default is None.

    name : str, optional
        Name of the effect. The default is None.

    frame : {'rest', 'obs'}, optional
        Frame in which the effect applies: 'rest' for rest-frame, 'obs' for
        observer-frame. The default is None.

    model : dict, optional
        Model for the effect parameters. The default is {}.
    """
    def __init__(self, 
                 effect=None, 
                 name=None,
                 frame=None,
                 model={}):
        """Initialize the Effect class."""
        self._effect = effect
        self._name = name
        self._frame = frame
        self._model = model
    
    @classmethod
    def from_sncosmo(cls, effect, name, frame, model={}):
        """Load an effect from a `sncosmo` effect.

        Parameters
        ----------
        effect : sncosmo.PropagationEffect
            Propagation effect.

        name : str
            Name of the effect.

        frame : {'rest', 'obs'}
            Frame in which the effect applies: 'rest' for rest-frame, 'obs'
            for observer-frame.

        model : dict, optional
            Model for the effect parameters. The default is {}.

        Returns
        -------
        Effect
            The loaded effect.
        """
        return cls(effect, name, frame, model=model)
    
    @classmethod
    def from_name(cls, name, which=None):
        """Load an effect from its name.

        Parameters
        ----------
        name : str
            Name of the effect (case insensitive, underscores ignored).
            Could be: 'mw', 'hostdust', 'scatter'.

        which : str, optional
            Which model to use for the effect:

            - for mw: 'ccm89' (default)
            - for hostdust: 'ccm89' (default)
            - for scatter: 'g10' or 'c11' (required)

            The default is None.

        Returns
        -------
        Effect
            The loaded effect.

        Raises
        ------
        NotImplementedError
            If `name` is not recognized or the dust law is not 'ccm89'.

        ValueError
            If the color scatter model `which` is unknown.
        """
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
                
            name = "host"
            frame = "rest"
            model = hostdust.dust_model

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
        """String representation of the effect."""        
        return self.__str__()
    
    def __str__(self):
        """String representation of the effect."""
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
        """Propagation effect (sncosmo.PropagationEffect)."""
        return self._effect
    
    @property
    def name(self):
        """Name of the effect."""
        return self._name
    
    @property
    def frame(self):
        """Frame of the effect ('rest' or 'obs')."""
        return self._frame

    @property
    def model(self):
        """Model of the effect parameters."""
        return self._model
