""" module to help SNANA users """
import numpy as np
import pandas

__all__ = ["read_simlib"]

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
    columns = [l_strip.lower() for l in block[read_start+1].replace("#", "").split()
              if len(l_strip:=l.strip())>1]

    # data
    data_block = block[read_start+2:]
    data = []
    for block_line in data_block:
        try:
            data_, comments = block_line.split("#")
        except:
            print(block_line)
            return
        case, data_ = data_.split(":")
        data_ = data_.split()
        data.append([case]+data_+[comments.strip()])
    
    dataframe = pandas.DataFrame(data, columns=["case"]+columns+["comments"])

    # metadata
    try:
        meta_block = block[:read_start]
        meta = " ".join([l.split("#")[0] for l in meta_block if not l.startswith("#") and len(l)>0 
                         and not "LIBGEN" in l]
               ).replace(": ", ":").split()
        meta = pandas.Series({k.lower():v for k,v in [l.split(":") for l in meta]})
    except:
        print(f"failed meta for {meta_block}")
        meta= None
    return dataframe, meta


### DES ####
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
    columns = [l_strip.lower() for l in block[read_start+1].replace("#", "").split()
              if len(l_strip:=l.strip())>1]

    # data
    data_block = block[read_start+2:]
    data = []
    for block_line in data_block:
        try:
            data_, comments = block_line.split("#")
        except:
            print(block_line)
            return
        case, data_ = data_.split(":")
        data_ = data_.split()
        data.append([case]+data_+[comments.strip()])
    
    dataframe = pandas.DataFrame(data, columns=["case"]+columns+["comments"])

    # metadata
    try:
        meta_block = block[:read_start]
        meta = " ".join([l.split("#")[0] for l in meta_block if not l.startswith("#") and len(l)>0 
                         and not "LIBGEN" in l]
               ).replace(": ", ":").split()
        meta = pandas.Series({k.lower():v for k,v in [l.split(":") for l in meta]})
    except:
        print(f"failed meta for {meta_block}")
        meta= None
    return dataframe, meta
            
