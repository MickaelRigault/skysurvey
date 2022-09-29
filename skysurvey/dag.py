import numpy as np
import pandas
import inspect
import warnings


class ModelDAG( object ):
    """
    Models are dict of arguments that may have 4 entries:
    {
    "element_1":{model:func, param: dict, as: str_or_list, input: list_of_former_element},
    "element_2":{model:func, param: dict, as: str_or_list, input: list_of_former_element},
    "element_3":{model:func, param: dict, as: str_or_list, input: list_of_former_element},
    ...
    }
    
    """
    def __init__(self, model, obj=None):
        """ 
        
        Parameters
        ----------
        model: dict
            dictionary descripting the model DAG

        obj: object
            instance the model is attached too. It may contain 
            method called by the DAG.

        Returns
        -------
        instance
        """
        self.model = model
        self.obj = obj

    def __str__(self):
        """ """
        import pprint
        return pprint.pformat(self.model, sort_dicts=False)

    def __repr__(self):
        """ """
        return self.__str__()

    # ============ #
    #   Method     #
    # ============ #
    def get_model(self, **kwargs):
        """ get a copy of the model 
        
        Parameters
        ----------

        **kwargs can change the model entry parameters
            for istance, t0: {"low":0, "high":10}
            will update model["t0"]["param"] = ...

        Returns
        -------
        dict
           a copy of the model (with param potentially updated)
        """
        model = self.model
        for k,v in kwargs.items():
            model[k]["param"].update(v)

        return model
    
    def change_model(self, **kwargs):
        """ change the model attached to this instance
        
        **kwargs will update the entry  parameters ("param", e.g. t0["param"])

        See also
        --------
        get_model: get a copy of the model
        """
        self.model = self.get_model(**kwargs)
        
    # ============ #
    #  Drawers     #
    # ============ #
    def draw(self, size=None, **kwargs):
        """ draw a random sampling of the parameters following
        the model DAG

        Parameters
        ----------
        size: int
            number of elements you want to draw.

        **kwargs goes to get_model()

        Returns
        -------
        pandas.DataFrame
            N=size lines and at least 1 column per model entry
        """
        model = self.get_model(**kwargs)
        return self._draw(model, size=size)
    
    def draw_param(self, name=None, model=None, size=None, xx=None, **kwargs):
        """ draw a single entry of the model

        Parameters
        ----------
        name: str
            name of the variable
            
        model_func: str, function
            what model should be used to draw the parameter

        size: int
            number of line to be draw

        xx: str, array
           provide this *if* the model_func returns the pdf and not sampling.
           xx defines where the pdf will be evaluated.
           if xx is a string, it will be assumed to be a np.r_ entry (``np.r_[xx]``)

        Returns
        -------
        list 
            
        """
        # Flexible origin of the sampling method
        func = self._parse_inputmodel_func(name=name, model=model)
        
        # Check the function parameters
        try:
            func_arguments = list(inspect.getfullargspec(func).args)
        except: # fail for Cython functions
            #warnings.warn(f"inspect failed for {name}{model} -> {func}")
            func_arguments = ["size"] # let's assume this as for numpy.random or scipy.

        # And set the correct parameters
        prop = {}
        if "size" in func_arguments:
            prop["size"] = size
            
        if "model" in func_arguments and model is not None: # means you left the default
            prop["model"] = model

        if "xx" in func_arguments and xx is not None: # if xx is None
            if type(xx) == str: # assumed r_ input
                xx = eval(f"np.r_[{xx}]")
                
            prop["xx"] = xx

        # Draw it.
        draw_ = func(**{**prop, **kwargs})
        if "xx" in func_arguments: # draw_ was actually a pdf
            xx_, pdf_ = draw_
            draw_ = self.draw_from_pdf(pdf_, xx_, size)
            
        return draw_
            
    @staticmethod
    def draw_from_pdf(pdf, xx, size):
        """ randomly draw from xx N=size elements following the pdf. """
        if type(xx) == str: # assumed r_ input
            xx = eval(f"np.r_[{xx}]")

        return np.random.choice(xx, size=size, p=pdf/pdf.sum())

    def _draw(self, model, size=None):
        """ core method converting model into a DataFrame (interp) """
        interp = pandas.DataFrame()
        params = dict(size=size)
        
        for param_name, param_model in model.items():
            # Default kwargs given
            if (inprop := param_model.get("param", {})) is None:
                inprop = {}

            # set the model ; this overwrite prop['model'] but that make sense.
            inprop["model"] = param_model.get("model", None)

            # read the parameters
            if (inparam := param_model.get("input", None)) is not None:
                for k in inparam:
                    if k in interp:
                        inprop[k] = interp[k].values
                    elif hasattr(self,k):
                        inprop[k] = getattr(self, k)
                    else:
                        raise ValueError(f"cannot get the input parameter {k}")

            # update the general properties for that of this parameters
            prop = {**params, **inprop}
            # 
            # Draw it
            samples = np.asarray(self.draw_param(param_name, **prop))
            
            # and feed
            output_name = param_model.get("as", param_name)
            interp[output_name] = samples.T
            
        return interp
    
    def _parse_inputmodel_func(self, name=None, model=None):
        """ returns the function associated to the input model.

        """
        if callable(model): # model is a function. Good to go.
            return model
        
        # model is a method of this instance ?
        if model is not None and hasattr(self, model):
            func = getattr(self, model)
            
        # model is a method a given object ?
        elif model is not None and hasattr(self.obj, model):
            func = getattr(self.obj, model)
            
        # name is a draw_ method of this instance object ?            
        elif hasattr(self, f"draw_{name}"):
            func = getattr(self,f"draw_{name}")

        # name is a draw_ method of this instance object ?            
        elif hasattr(self.obj, f"draw_{name}"):
            func = getattr(self.obj, f"draw_{name}")
        
        else:
            try:
                func = eval(model) # if you input a string of a function known by python somehow ?
            except:
                raise ValueError(f"Cannot parse the input function {name}:{model}")
        
        return func
