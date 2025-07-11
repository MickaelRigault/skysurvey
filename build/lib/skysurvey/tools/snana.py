""" module to help SNANA users """
import numpy as np
import pandas

__all__ = ["read_simlib"]

def read_simlib(filepath):
    """ reads a simlib file and returns the 'block' elements

    Parameters
    ----------
    filepath: str or path
        file path to be opened with open(filepath,"r").

    Returns
    -------
    list
       meta, list_of_dataframe
       - meta: dataframe containing the metadata of all blocks
       - list_of_dataframe: list of all the block data.
    """
    file_ = open(filepath,"r").read().splitlines()
    blocks = []
    singleblock = []
    for f_ in file_:
        if f_.startswith("BEGIN LIBGEN"):
            singleblock = []
        elif f_.startswith("END_LIBID"):
            blocks.append(_parse_simlib_block(singleblock))
            singleblock = []
            continue

        singleblock.append(f_)


    dtypes = {**{k:"float32" for k in ["RA","DEC", "MWEBV","PIXSIZE",]},
              **{k:"int16" for k in ["NOBS","CCDNUM","LIBID"]}}
    metas = pandas.concat([b[1] for b in blocks], axis=1).T.astype(dtypes)
    dfs = [b[0] for b in blocks]
    return metas, dfs

def _parse_simlib_block(singleblock):
    """ internal function of read_simlib that parses a 'block' """
    # meta data
    meta_single = {l.split(":")[0]:l.split(":")[1].strip() 
                   for l in singleblock if not l.startswith("S:") and l.count(":")==1}
    meta_multi = {ls_.split(":")[0]:ls_.split(":")[1].strip() 
                  for l in singleblock if not l.startswith("S:") and l.count(":")>1 
                  for l_ in l.split('  ') if len(ls_:=l_.strip())>0}
    meta = {**meta_multi, **meta_single}
    meta
    
    # DataBlock
    columns = ["mjd","expid","filter", "gain","ccdnoise", "skysig", "psf1","psf2","psf21","zp","zp_err","mag"]
    dtypes = {k:"float32" for k in columns}
    dtypes["expid"] = "int16"
    dtypes["filter"] = "string"
    df = pandas.DataFrame([[ss.strip() for ss in s.replace("S:","").split()] for s in singleblock if s.startswith("S:")],
                         columns=columns).astype(dtypes)

    # NEA * skysig | see Section 6 of Kessler et al. 2019
    # and sigma_psf = np.sqrt( NEA/(4pi) ) -> NEA = sigma_psf**2 * 4pi
    df["skynoise"] = (df["psf1"]**2 * np.pi*4) * df["skysig"]
    return df, pandas.Series(meta)
