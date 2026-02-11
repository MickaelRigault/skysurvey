""" module to help SNANA users """
import numpy as np
import pandas
import warnings

__all__ = ["parse_simlib"]

def parse_simlib(simlib):
    """ """
    file_ = open(simlib, "r").read().splitlines()
    i_start = [ i for i, f_ in enumerate(file_) if f_.startswith("BEGIN LIBGEN") ]
    i_end = [ i for i, f_ in enumerate(file_) if f_.startswith("END_LIBID") ]
    if len(i_start) != 1:
        raise ValueError("Exactly 1 'BEGIN LIBGEN' is expected, {len(i_start)} found.")

    dfs = []
    metas = []
    blocks = i_start+i_end
    for block_range in zip(blocks[:-1], blocks[1:]):
        block = file_[block_range[0]:block_range[1]]
        df, meta = parse_simlib_block(block)
        dfs.append(df)
        metas.append(meta)

    data = pandas.concat(dfs, keys=np.arange(len(dfs)))
    metadata = pandas.concat(metas, keys=np.arange(len(dfs)), axis=1).T
    return data, metadata

def parse_simlib_block(block):
    """ """
    read_start = [ i for i, f_ in enumerate(block) if " READ " in f_]
    if len(read_start) == 0:
        raise ValueError("cannot parse input block. No 'READ' line found")
    if len(read_start) > 1:
        raise ValueError(f"cannot parse input block. multiple 'READ' lines found {read_start}")

    # ok this is the line with READ on it.
    read_start = read_start[0]
    # columns
    columns = [block_strip.lower() for block_ in block[read_start+1].replace("#", "").split()
              if len(block_strip:=block_.strip())>1]

    # data
    data_block = block[read_start+2:]
    data = []
    for block_line in data_block:
        try:
            data_, comments = block_line.split("#")
        except Exception as e:
            warnings.warn(e)
            print(block_line)
            return
        case, data_ = data_.split(":")
        data_ = data_.split()
        data.append([case]+data_+[comments.strip()])
    
    dataframe = pandas.DataFrame(data, columns=["case"]+columns+["comments"])

    # metadata
    try:
        meta_block = block[:read_start]
        meta = " ".join([meta_.split("#")[0] for meta_ in meta_block
                             if not meta_.startswith("#") and len(meta_)>0 
                         and "LIBGEN" not in meta_]
               ).replace(": ", ":").split()
        meta = pandas.Series({k.lower():v for k,v in [meta_.split(":") for meta_ in meta]})
    except Exception as e:
        warnings.warn(e)        
        print(f"failed meta for {meta_block}")
        meta= None
    return dataframe, meta


### DES ####
def parse_simlib_des(simlib):
    """ """
    file_ = open(simlib, "r").read().splitlines()
    i_start = [ i for i, f_ in enumerate(file_) if f_.startswith("BEGIN LIBGEN") ]
    i_end = [ i for i, f_ in enumerate(file_) if f_.startswith("END_LIBID") ]
    if len(i_start) != 1:
        raise ValueError("Exactly 1 'BEGIN LIBGEN' is expected, {len(i_start)} found.")

    dfs = []
    metas = []
    blocks = i_start+i_end
    for block_range in zip(blocks[:-1], blocks[1:]):
        block = file_[block_range[0]:block_range[1]]
        df, meta = parse_simlib_block(block)
        dfs.append(df)
        metas.append(meta)

    data = pandas.concat(dfs, keys=np.arange(len(dfs)))
    metadata = pandas.concat(metas, keys=np.arange(len(dfs)), axis=1).T
    return data, metadata

def parse_simlib_block_des(block):
    """ """
    read_start = [ i for i, f_ in enumerate(block) if " READ " in f_]
    if len(read_start) == 0:
        raise ValueError("cannot parse input block. No 'READ' line found")
    if len(read_start) > 1:
        raise ValueError(f"cannot parse input block. multiple 'READ' lines found {read_start}")

    # ok this is the line with READ on it.
    read_start = read_start[0]
    # columns
    columns = [l_strip.lower() for line_ in block[read_start+1].replace("#", "").split()
              if len(l_strip := line_.strip())>1]

    # data
    data_block = block[read_start+2:]
    data = []
    for block_line in data_block:
        try:
            data_, comments = block_line.split("#")
        except Exception as e:
            warnings.warn(e)
            print(block_line)
            return
        
        case, data_ = data_.split(":")
        data_ = data_.split()
        data.append([case]+data_+[comments.strip()])
    
    dataframe = pandas.DataFrame(data, columns=["case"]+columns+["comments"])

    # metadata
    try:
        meta_block = block[:read_start]
        meta = " ".join([block_.split("#")[0] for block_ in meta_block
                             if not block_.startswith("#") and len(block_)>0 
                         and "LIBGEN" not in block_]
               ).replace(": ", ":").split()
        meta = pandas.Series({k.lower():v for k,v in [meta_.split(":") for meta_ in meta]})
    except Exception as e:
        warnings.warn(e)
        print(f"failed meta for {meta_block}")
        meta= None
    return dataframe, meta
            
