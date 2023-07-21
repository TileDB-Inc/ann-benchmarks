from __future__ import absolute_import
import numpy
import os
import tiledb

from tiledb.vector_search.ingestion import ingest
from tiledb.vector_search.index import IVFFlatIndex
import numpy as np
import multiprocessing


from ..base.module import BaseANN

class TileDBIVFFlat(BaseANN):
    def __init__(self, metric, n_list):
        self._n_list = n_list
        self._metric = metric

    def query(self, v, n):
        if self._metric == 'angular':
            raise NotImplementedError()
        self.res =self.index.query(np.array([v]).astype(numpy.float32), k=n, nthreads=multiprocessing.cpu_count(), nprobe=min(self._n_probe,self._n_list), use_nuv_implementation=True)
        return self.res[0]

    def batch_query(self, X, n):
        if self._metric == 'angular':
            raise NotImplementedError()
        self.res =self.index.query(X.astype(numpy.float32), k=n, nthreads=multiprocessing.cpu_count(), nprobe=min(self._n_probe, self._n_list), use_nuv_implementation=True)

    def get_batch_results(self):
        return self.res

    def fit(self, X):
        source_uri = "/tmp/data"
        array_uri = "/tmp/array"
        if os.path.isfile(source_uri):
            os.remove(source_uri)
        if os.path.isfile(array_uri):
            os.remove(array_uri)
        f = open(source_uri, "wb")
        np.array(X.shape, dtype="uint32").tofile(f)
        X.tofile(f)
        f.close()

        if X.dtype == "uint8":
            source_type = "U8BIN"
        elif X.dtype == "float32":
            source_type = "F32BIN"
        maxtrain = min(50 * self._n_list, X.shape[0])
        self.index = ingest(index_type="IVF_FLAT",
                       array_uri=array_uri,
                       source_uri=source_uri,
                       source_type=source_type,
                       size=X.shape[0],
                       training_sample_size=maxtrain,
                       partitions=self._n_list,
                       input_vectors_per_work_item=100000000,
                     )
        self.index = IVFFlatIndex(uri=array_uri, dtype=X.dtype, memory_budget=-1)

    def set_query_arguments(self, n_probe):
        self._n_probe = n_probe

    def get_additional(self):
        return {}

    def __str__(self):
        return 'TileDBIVFFlat(n_list=%d, n_probe=%d)' % (self._n_list, self._n_probe)
