# # Mamba approach
# FROM continuumio/miniconda3:4.10.3

# RUN conda install mamba -n base -c conda-forge

# RUN mamba install -y -c tiledb tiledb tiledb-vector-search

# WORKDIR /home/app
# COPY requirements.txt run_algorithm.py ./
# RUN pip3 install -r requirements.txt
# ENTRYPOINT ["python", "-u", "run_algorithm.py"]

# # Conda approach
# FROM ann-benchmarks

# RUN apt update && apt install -y wget
# RUN wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
# RUN bash Anaconda3-2020.11-Linux-x86_64.sh -b

# ENV PATH /root/anaconda3/bin:$PATH

# RUN conda install -y -c tiledb tiledb
# RUN conda install -y -c conda-forge tiledb-vector-search

# WORKDIR /home/app
# COPY requirements.txt run_algorithm.py ./
# RUN pip3 install -r requirements.txt
# ENTRYPOINT ["python", "-u", "run_algorithm.py"]

# Pip approach
FROM ann-benchmarks

RUN pip install tiledb
RUN pip install tiledb-vector-search
RUN python3 -c 'import tiledb.vector_search'