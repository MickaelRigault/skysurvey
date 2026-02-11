import itertools
import pandas
import numpy as np


def isin_pair_elements(elements, test_elements):
    """
    Test whether each pair of integers in elements is present in test_elements.

    Parameters
    ----------
    elements: array_like 
        Array of integer pairs to test.

    test_element: array_like
        Array of integer pairs defining the reference set.

    Returns
    -------
    isin: ndarray, bool
        Boolean array. True if the corresponding pair in elements is present in test_elements, False otherwise.
    """
    elements_combined = (elements[:, 0] << 16) | elements[:, 1]
    test_elements_combined = (test_elements[:, 0] << 16) | test_elements[:, 1]
    return np.isin(elements_combined, test_elements_combined)

# pandas concat tricks suggested by: AntoineGillesLordet (https://github.com/MickaelRigault/skysurvey/issues/35)
# aranged by: Mickael Rigault
def chunk_dfs(dfs, chunk_size):
    """
    Split an iterable of DataFrames into successive chunks.

    Parameters
    ----------
    dfs: iterable of pandas.DataFrame
        Iterable yielding DataFrames to be grouped into chunks.

    chunk_size: int
        Number of DataFrames per chunk.

    Yields
    -------
    chunk : list of pandas.DataFrame
        List of DataFrames in the current chunk.
    size : int
        Number of DataFrames in the chunk (may be smaller than chunk_size for the last chunk).  
    """
    dfs_out = []
    for df in dfs:
        dfs_out.append(df)
        if len(dfs_out) == chunk_size:
            yield dfs_out, chunk_size
            dfs_out = []
            
    if dfs_out:
        yield dfs_out, len(dfs_out)

def concat_chunk(dfs, **kwargs):
    """
    Concatenate a chunk of DataFrames using pandas.concat.
    
    Parameters
    ----------
    dfs: iterable of pandas.DataFrame
        DataFrames to concatenate.

    **kwargs
        Additional keyword arguments passed to pandas.concat.

    Returns
    -------
    pandas.DataFrame
        Concatenated DataFrame.
    """
    return pandas.concat((df for df in dfs), **kwargs)

def eff_concat(dfs, chunk_size, keys=None, **kwargs):
    """  
    Efficiently concatenate a large number of DataFrames by chunking.
    
    Parameters
    ----------
    dfs: iterable of pandas.DataFrame
        DataFrames to concatenate.

    chunk_size: int
        Number of DataFrames per chunk.
    
    keys : sequence, optional
        Keys to use for indexing, passed to pandas.concat.
        When chunking, the corresponding slice of keys is passed to each chunk. Default is None.

    **kwargs
        Additional keyword arguments passed to pandas.concat.
    
    Returns
    -------
    pandas.DataFrame
        Concatenated DataFrame.
    """
    dfs, dfs_len = itertools.tee(dfs, 2)
    if len(list(dfs_len)) < chunk_size:
        return concat_chunk(dfs, keys=keys, **kwargs)
    
    if keys is None:
        return pandas.concat(concat_chunk(dfs_chunk, **kwargs)
            for dfs_chunk, _ in chunk_dfs(dfs, chunk_size)
        )

    return pandas.concat( (concat_chunk(dfs, keys=keys[i*chunk_size:i*chunk_size+step_], **kwargs)
                            for i, (dfs, step_) in enumerate( chunk_dfs(dfs, chunk_size))
                          )
                        )