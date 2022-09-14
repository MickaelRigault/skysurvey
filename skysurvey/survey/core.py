import warnings
import numpy as np
import pandas

class Survey( object ):
    
    REQUIRED_COLUMNS = ['mjd', 'band', 'skynoise', 'fieldid', "gain", "zp"]
    
    def __init__(self, data):
        """ 
        See also
        --------
        
        """
        self.set_data(data)
    
    def __array__(self):
        """ """
        return self.data.__array__()
    
    # ============== #
    #   Methods      #
    # ============== #    
    def set_data(self, data):
        """ set the observing data 

        = It is unlikely you need to use that directly. =

        Parameters
        ----------
        data: pandas.DataFrame
             observing data. see REQUIRED_COLUMNS for the list of
             required columns.

        Returns
        -------
        None
        """
        if data is not None and not np.in1d(self.REQUIRED_COLUMNS, data.columns).all():
            raise ValueError(f"at least one of the following column name if missing {self.REQUIRED_COLUMNS}")
        self._data = data
        
    # ------------ #
    #   GETTER     #
    # ------------ #
    def get_fieldcoverage(self, incl_zeros=False, fillna=np.NaN,
                          **kwargs):
        """ short cut to get_fieldstat('size') 

        Parameters
        ----------
        incl_zeros: bool
            fields will no entries will not be shown 
            except if incl_zeros is True

        fillna: float, str
            format of the N/A entries

        **kwargs goes to get_fieldstat()

        Returns
        -------
        DataFrame or Serie 
            following groupby.agg()

        See also
        --------
        get_fieldstat: get observing statistics for the fields

        """
        return self.get_fieldstat(stat="size", columns=None,
                                    incl_zeros=incl_zeros, 
                                  fillna=fillna, **kwargs)
    
    def get_fieldstat(self, stat, columns=None,
                        incl_zeros=False, fillna=np.NaN,
                        data=None):
        """ get observing statistics for the fields

        basically a shortcut to ``data.groupby("fieldid")[`column`].`stat`()`` 
        
        Parameters
        ----------
        stat: str, list
            element to be passed to groupby.agg() 
            could be e.g.: 'mean' or ['mean', 'std'] or [np.median, 'mean'] etc.
            If stat = 'size', this returns data["fieldid"].value_counts()
                (slightly faster than groupby("fieldid").size()).
                
        columns: str, list, None
            name of the columns to be kept.
            None means no cut.

        incl_zeros: bool
            fields will no entries will not be shown 
            except if incl_zeros is True

        fillna: float, str
            format of the N/A entries
            
        data: pandas.DataFrame, None
            data you want this to be applied to.
            if None, a copy of self.data is used.
            = leave to None if unsure =
        
        Returns
        -------
        DataFrame or Serie 
            following groupby.agg()

        """
        if data is None:
            data = self.data.copy()
            
        if stat in ["size","value_counts"]:
            data = data["fieldid"].value_counts()
            
        elif columns is None:
            data = data.groupby("fieldid").agg(stat)
        else:
            data = data.groupby("fieldid")[columns].agg(stat)
            
        if not incl_zeros:
            return data
        
        if type(data) == pandas.Series: # Serie
            all_data = np.ones(self.nfields)*fillna
            all_data[data.index] = data.values
            all_data = pandas.Series(all_data, name=f"fieldid_{stat}")
            
        else: # DataFrame
            all_data = np.ones((self.nfields, data.shape[1]))*fillna
            all_data[data.index] = data.values
            all_data = pandas.DataFrame(all_data, columns=data.columns)
            
        return all_data
        
        
    def radec_to_fieldid(self, ra, dec):
        """ get the fieldid of the given (list of) coordinates """
        raise NotImplementedError("you have not implemented radec_to_fieldid for your survey")
    
    # ----------- #
    #  PLOTTER    #
    # ----------- #        
    def show(self):
        """ shows the sky coverage """
        raise NotImplementedError("you have not implemented show for your survey")
        
    # ============== #
    #   Properties   #
    # ============== #
    @property
    def data(self):
        """ dataframe containing what has been observed when
        aka. the observing data 
        """
        return self._data
    
    @property
    def metadata(self):
        """ metadata associated to the survey """
        meta = {"type":self.of_type}
        return meta
    
    @property    
    def nfields(self):
        """ number of fields """
        if not hasattr(self,"_nfields") or self._nfields is None:
            warnings.warn("no nfields set, so this is assuming max of data['fieldid'].")
            self._nfields =self.data["fieldid"].max()
            
        return self._nfields
    
    @property
    def of_type(self):
        """ kind of survey that is """
        return str(type(self)).split("'")[-2].split(".")[-1]

