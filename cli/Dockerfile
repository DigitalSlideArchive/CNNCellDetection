
FROM cramraj8/base_docker_cuda:1.0
# FROM dsarchive/histomicstk:latest
MAINTAINER Lee Cooper <lee.cooper@emory.edu>


# Copy plugin files and install any requirements
ENV my_cwd_path=$htk_path/../my_cwd
RUN mkdir -p $my_cwd_path && \
	apt-get update


COPY . $my_cwd_path

# ================ Above are basics ======================

RUN pip install -e git+https://github.com/cramraj8/luminoth.git@nuclei-detection#egg=luminoth --ignore-installed PyYAML

RUN pip install -r $htk_path/requirements.txt

RUN pip install tensorflow==1.5.0

# ================ Above are installation requirements file ======================

# use entrypoint provided by slicer_cli_web
WORKDIR $my_cwd_path
ENTRYPOINT ["/build/miniconda/bin/python" ,"/build/slicer_cli_web/server/cli_list_entrypoint.py"]
