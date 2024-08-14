FROM ann-benchmarks

# Option 1: Install pre-built library.
RUN pip install tiledb tiledb-vector-search

# Option 2: Build the library ourselves.
# RUN git clone https://github.com/TileDB-Inc/TileDB-Vector-Search.git
# RUN cd TileDB-Vector-Search &&  pip install .

RUN python3 -c 'import tiledb.vector_search'