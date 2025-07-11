
# Suggested by: AntoineGillesLordet (https://github.com/MickaelRigault/skysurvey/issues/35)
# Aranged by: Mickael Rigault
import itertools
import pandas

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
    
    return pandas.concat( (concat_chunk(dfs, keys=keys[i*chunk_size:i*chunk_size+l], **kwargs)
                            for i, (dfs, l) in enumerate( chunk_dfs(dfs, chunk_size))
                          )
                        )
