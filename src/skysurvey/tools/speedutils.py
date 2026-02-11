

import itertools
import pandas
import numpy as np


def isin_pair_elements(elements, test_elements):
    """ """
    elements_combined = (elements[:, 0] << 16) | elements[:, 1]
    test_elements_combined = (test_elements[:, 0] << 16) | test_elements[:, 1]
    return np.isin(elements_combined, test_elements_combined)

# pandas concat tricks suggested by: AntoineGillesLordet (https://github.com/MickaelRigault/skysurvey/issues/35)
# aranged by: Mickael Rigault
def chunk_dfs(dfs, chunk_size):
    """ """
    dfs_out = []
    for df in dfs:
        dfs_out.append(df)
        if len(dfs_out) == chunk_size:
            yield dfs_out, chunk_size
            dfs_out = []
            
    if dfs_out:
        yield dfs_out, len(dfs_out)

def concat_chunk(dfs, **kwargs):
    """ """
    return pandas.concat((df for df in dfs), **kwargs)

def eff_concat(dfs, chunk_size, keys=None, **kwargs):
    """ """
    dfs, dfs_len = itertools.tee(dfs, 2)
    if len(list(dfs_len)) < chunk_size:
        return concat_chunk(dfs, keys=keys, **kwargs)
    
    return pandas.concat( (concat_chunk(dfs, keys=keys[i*chunk_size:i*chunk_size+step_], **kwargs)
                            for i, (dfs, step_) in enumerate( chunk_dfs(dfs, chunk_size))
                          )
                        )
