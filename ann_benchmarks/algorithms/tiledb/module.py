import os
import numpy as np
import multiprocessing

from ..base.module import BaseANN

from tiledb.vector_search.ingestion import ingest
from tiledb.vector_search import IVFFlatIndex
from tiledb.vector_search import IVFPQIndex
from tiledb.vector_search import FlatIndex
from tiledb.vector_search import VamanaIndex
from tiledb.cloud.dag import Mode

from ..base.module import BaseANN

MAX_UINT64 = np.iinfo(np.dtype("uint64")).max

class TileDB(BaseANN):
    def __init__(self, metric, index_type, n_list = -1, l_build = -1, r_max_degree = -1, num_subspaces_divisor = -1):
        self._index_type = index_type
        self._metric = metric
        self._n_list = n_list
        self._l_build = l_build
        self._r_max_degree = r_max_degree
        self._num_subspaces_divisor = num_subspaces_divisor
        self._n_probe = -1
        self._l_search = -1

    def query(self, v, n):
        if self._metric == 'angular':
            raise NotImplementedError()

        # query() returns a tuple of (distances, ids).
        ids = self.index.query(
            np.array([v]).astype(np.float32), 
            k=n, 
            nthreads=multiprocessing.cpu_count(), 
            nprobe=min(self._n_probe, self._n_list), 
            l_search=self._l_search
        )[1][0]
        # Fix for 'OverflowError: Python int too large to convert to C long'.
        ids[ids == MAX_UINT64] = 0
        return ids 

    def batch_query(self, X, n):
        if self._metric == 'angular':
            raise NotImplementedError()
        # query() returns a tuple of (distances, ids).
        self.res = self.index.query(
            X.astype(np.float32), 
            k=n, 
            nthreads=multiprocessing.cpu_count(), 
            nprobe=min(self._n_probe, self._n_list), 
            l_search=self._l_search
        )[1]
        # Fix for 'OverflowError: Python int too large to convert to C long'.
        self.res[self.res == MAX_UINT64] = 0

    def get_batch_results(self):
        return self.res

    def fit(self, X):
        array_uri = "/tmp/array"
        if os.path.isfile(array_uri):
            os.remove(array_uri)

        dimensions = X.shape[1]
        self.index = ingest(
            index_type=self._index_type,
            index_uri=array_uri,
            input_vectors=X,
            partitions=self._n_list,
            l_build=self._l_build,
            r_max_degree=self._r_max_degree,
            num_subspaces=dimensions/self._num_subspaces_divisor
        )
        if self._index_type == "IVF_FLAT":
            self.index = IVFFlatIndex(uri=array_uri)
        elif self._index_type == "IVF_PQ":
            self.index = IVFPQIndex(uri=array_uri)
        elif self._index_type == "FLAT":
            self.index = FlatIndex(uri=array_uri)
        elif self._index_type == "VAMANA":
            self.index = VamanaIndex(uri=array_uri)
        else:
            raise ValueError(f"Unsupported index {self._index_type}")

    def get_additional(self):
        return {}

class TileDBIVFFlat(TileDB):
    def __init__(self, metric, n_list):
        super().__init__(
            index_type="IVF_FLAT",
            metric=metric,
            n_list=n_list,
        )
    
    def set_query_arguments(self, n_probe):
        self._n_probe = n_probe

    def __str__(self):
        return 'TileDBIVFFlat(n_list=%d, n_probe=%d)' % (self._n_list, self._n_probe)

class TileDBFlat(TileDB):
    def __init__(self, metric, _):
        super().__init__(
            index_type="FLAT",
            metric=metric
        )
    
    def __str__(self):
        return 'TileDBFlat()'

class TileDBVamana(TileDB):
    def __init__(self, metric, r_max_degree):
        super().__init__(
            index_type="VAMANA",
            metric=metric,
            l_build=60,
            r_max_degree=r_max_degree
        )
    
    def set_query_arguments(self, l_search):
        self._l_search = l_search
    
    def __str__(self):
        return 'TileDBVamana(l_build=%d, r_max_degree=%d, l_search=%d)' % (self._l_build, self._r_max_degree, self._l_search)

class TileDBIVFPQ(TileDB):
    def __init__(self, metric, n_list, num_subspaces_divisor):
        super().__init__(
            index_type="IVF_PQ",
            metric=metric,
            n_list=n_list,
            num_subspaces_divisor=num_subspaces_divisor
        )
    
    def set_query_arguments(self, n_probe):
        self._n_probe = n_probe

    def __str__(self):
        return 'TileDBIVFPQ(n_list=%d, n_probe=%d, num_subspaces_divisor=%d)' % (self._n_list, self._n_probe, self._num_subspaces_divisor)