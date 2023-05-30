import numpy as np
import pandas
import inspect
import warnings



def modeldict_to_modeldf(model):
    """ """
    dd = pandas.DataFrame(list(model.values()), 
                          index=model.keys()).reset_index(names=["model_name"])
    # naming convention
    f_ = dd["as"].fillna(dict(dd["model_name"]))
    f_.name = "entry"
    # merge and explode the names and inputs
    return dd.join(f_)


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

    def get_backward_entries(self, name, incl_input=True):
        """ get the list of entries that affects the input on.
        Changing any of the return entry name impact the given name.

        Parameters
        ----------
        name: str
            name of the entry

        incl_input: bool
            should the returned list include or not 
            the given name ? 

        Return
        ------
        list
            list of backward entry names 
        """
        depends_on = self.entry_dependencies.dropna()

        names = np.atleast_1d(name)
        if incl_input:
            backward_entries = list(names)
        else:
            backward_entries = []

        leads_to_changing = depends_on.loc[depends_on.index.isin(names)]
        while len(leads_to_changing)>0:
            _ = [backward_entries.append(name_) for name_ in list(leads_to_changing.values)]
            leads_to_changing = depends_on.loc[depends_on.index.isin(leads_to_changing)]

        return backward_entries

    def get_forward_entries(self, name, incl_input=True):
        """ get the list of forward entries. 
        These would be affected if the given entry name is changed.

        Parameters
        ----------
        name: str
            name of the entry

        incl_input: bool
            should the returned list include or not 
            the given name ? 

        Return
        ------
        list
            list of forward entry names 
        """
        inputs_of = self.entry_inputof.explode()

        names = np.atleast_1d(name)
        if incl_input:
            forward_entries = list(names)
        else:
            forward_entries = []

        leads_to_changing = inputs_of.loc[inputs_of.index.isin(names)]
        while len(leads_to_changing)>0:
            # all names individually
            _ = [forward_entries.append(name_) for name_ in list(leads_to_changing.values)]
            leads_to_changing = inputs_of.loc[inputs_of.index.isin(leads_to_changing)]

        return forward_entries


    def get_modeldf(self):
        """ """
        modeldf = modeldict_to_modeldf(self.model)
        return modeldf.explode("entry").explode("input").set_index("entry")
        
    # ============ #
    #  Drawers     #
    # ============ #
    def redraw_from(self, name, data, incl_name=True, size=None, **kwargs):
        """ re-draw the data starting from the given entry name.
        
        All forward values will be updated while the independent 
        entries are left unchanged.

        Parameters
        ----------
        name: str
            entry name. See self.entries

        data: pandas.DataFrame
            data to be updated 
            Must at least include entry needed by name if any. 
            See self.entry_dependencies

        incl_name: bool
            should the given name be updated or not ?

        size: None
            number of entries to draw. Ignored if not needed.

        **kwargs goes to self.draw() -> get_model

        Returns
        -------
        pandas.DataFrame
            the updated version of the input data.

        """
        limit_to_entries = self.get_forward_entries(name, incl_input=incl_name)
        return self.draw(size, limit_to_entries=limit_to_entries, data=data)

    
    def draw(self, size=None, limit_to_entries=None, data=None, **kwargs):
        """ draw a random sampling of the parameters following
        the model DAG

        Parameters
        ----------
        size: int
            number of elements you want to draw.


        limit_to_entries: list
            if given, entries not in this list will be ignored.
            see self.entries

        data: pandas.DataFrame
            starting point for the draw.

        **kwargs goes to get_model()

        Returns
        -------
        pandas.DataFrame
            N=size lines and at least 1 column per model entry
        """
        model = self.get_model(**kwargs)
        return self._draw(model, size=size, limit_to_entries=limit_to_entries,
                              data=data)
    
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

    def _draw(self, model, size=None, limit_to_entries=None, data=None):
        """ core method converting model into a DataFrame (interp) """
        if size == 0:
            columns = list(np.hstack([v.get("as", name) for name, v in model.items()]))
            return pandas.DataFrame(columns=columns)

        if data is None:
            data = pandas.DataFrame()
        else:
            data = data.copy() # make sure you are changing a copy
            
        params = dict(size=size)
        for param_name, param_model in model.items():
            if limit_to_entries is not None and param_name not in limit_to_entries:
                continue
            
            # Default kwargs given
            if (inprop := param_model.get("param", {})) is None:
                inprop = {}

            # set the model ; this overwrite prop['model'] but that make sense.
            inprop["model"] = param_model.get("model", None)

            # read the parameters
            if (inparam := param_model.get("input", None)) is not None:
                for k in inparam:
                    if k in data:
                        inprop[k] = data[k].values
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
            data[output_name] = samples.T
            
        return data
    
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

    # =================== #
    #   Properties        #
    # =================== #
    @property
    def entries(self):
        """ array of model entry names """
        modeldf = self.get_modeldf()
        return np.asarray(modeldf.index.unique(), dtype=str)

    @property
    def entry_dependencies(self):
        """ pandas series of entry input dependencies (exploded) | NaN is not entry """
        modeldf = self.get_modeldf()
        return modeldf["input"]

    @property
    def entry_inputof(self):
        """ """
        # maybe a bit overcomplicated...
        modeldf = self.get_modeldf()
        return modeldf[~modeldf["input"].isna()].reset_index().set_index("input")["entry"]
        

def get_modeldf(self):
    """ """
    modeldf = modeldict_to_modeldf(self.model)
    return modeldf.explode("name").explode("input")    
