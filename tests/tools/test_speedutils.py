import numpy as np
import pandas
from skysurvey.tools.speedutils import isin_pair_elements, chunk_dfs, concat_chunk, eff_concat

# tests for the func isinpair_elements
def test_isin_pair_elements():
    elements = np.array([[1,2], [3,4], [5,6] , [7,8]])
    test_elements = np.array([[3,4], [7,8], [9,10]])

    output = isin_pair_elements(elements, test_elements)
    output_expected = np.array([False, True, False, True])

    assert isinstance(output, np.ndarray)
    output.shape == len(elements)
    assert np.array_equal(output, output_expected)

# tests for the func chunk_dfs
def test_chunk_dfs():
    dfs = [pandas.DataFrame({"a": [i]}) for i in range(5)]
    chunk_size = 2

    chunks = list(chunk_dfs(dfs, chunk_size))
    sizes = [size for _, size in chunks]

    assert sizes == [2, 2, 1]
    assert len(chunks) == 3

def test_chunk_dfs_mutiple():
    dfs = [pandas.DataFrame({"a": [i]}) for i in range(4)]
    chunk_size = 2  

    chunks = list(chunk_dfs(dfs, chunk_size))
    sizes = [size for _, size in chunks]
    
    assert sizes == [2, 2]
    assert len(chunks) == 2

# tests for the func concat_chunk
def test_concat_chunk():
    df1 = pandas.DataFrame({"a": [1, 2]})
    df2 = pandas.DataFrame({"a": [3]})
    
    output = concat_chunk([df1, df2])
    output_expected = pandas.concat([df1, df2])
    pandas.testing.assert_frame_equal(output, output_expected)

# tests for the func eff_concat
def test_eff_concat_direct_concat():
    dfs = [pandas.DataFrame({"a": [1]}), pandas.DataFrame({"a": [2]})]
    chunk_size = 10

    output = eff_concat(dfs, chunk_size=chunk_size)
    output_expected = pandas.concat(dfs)
    pandas.testing.assert_frame_equal(output, output_expected)

def test_eff_concat_chunked_concat_keys_is_none():
    dfs = [pandas.DataFrame({"a": [i]}) for i in range(5)]
    chunk_size = 2

    output = eff_concat(dfs, chunk_size=chunk_size)
    output_expected = pandas.concat(dfs)
    pandas.testing.assert_frame_equal(output, output_expected)

def test_eff_concat_chunked_concat_with_keys():
    dfs = [pandas.DataFrame({"a": [i]}) for i in range(5)]
    keys = ["0", "1", "2", "3", "4"]
    chunk_size = 2

    output = eff_concat(dfs, chunk_size=chunk_size, keys=keys)
    output_expected = pandas.concat(dfs, keys=keys)
    pandas.testing.assert_frame_equal(output, output_expected)