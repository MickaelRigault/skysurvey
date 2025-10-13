
from shapely import geometry
import numpy as np
import pandas

def get_roman_footprint():
    """ 
    Get the Roman footprint.
    quick and dirty that have a 1011arcmin FoV

    Returns
    -------
    shapely.geometry.Polygon
    """
    roman_footprint = geometry.Polygon(
                    np.asarray([[0, 1.], [0.15, 1.],
                                  [0.15, 0.8],[0.3, 0.8],
                                  [0.3, 0.75],[0.70, 0.75],
                                  [0.70, 0.8],[0.85, 0.8],
                                  [0.85, 1.], [1,1.],
                                  [1, 0.25], [0.85, 0.25],
                                  [0.85, 0.1], [0.7, 0.1],
                                  [0.7, 0], [0.3, 0],
                                  [0.3, 0.1], [0.15, 0.1],
                                  [0.15, 0.25], [0., 0.25],
                                  [0,1.]
                                 ])*(1, 0.65)/1.304 # 0.281 deg^2
                )
    return roman_footprint


# ==================== #
#                      #
#     SIMLIB OFFICIAL  #
#                      #
# ==================== #
def parse_simlib(simlib):
    """ 

    Parameters
    ----------
    simlib: str, path
        the fullpath of a roman simlib file.

    Returns
    -------
    dataframe: pandas.DataFrame
        a multi-indexed dataframe (level=0 for simlib block index).

    metadata: pandas.DataFrame
        a dataframe of the block metadata.
    """
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
    """ official Roman simlib are built per block. This parses one.

    Parameters
    ----------
    block: list
        the list of data lines from the simlib.

    Returns
    -------
    dataframe: pandas.DataFrame
        the dataframe of the simlog

    meta: pandas.Series
        a serie containing the metadata of this block.
    """
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
            
