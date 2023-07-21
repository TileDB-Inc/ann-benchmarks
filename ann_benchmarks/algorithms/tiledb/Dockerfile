FROM tiledb_vs

RUN mamba install -y ansicolors docker-py h5py matplotlib numpy pyyaml psutil scipy scikit-learn jinja2 pandas

WORKDIR /home/app
COPY requirements.txt run_algorithm.py ./
RUN pip3 install -r requirements.txt
ENTRYPOINT ["python", "-u", "run_algorithm.py"]
