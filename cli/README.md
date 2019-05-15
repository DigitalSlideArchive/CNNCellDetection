
# Nuclei Detection Web CLI Plugin

An extended plugin for [girder/slicer_cli_web](https://github.com/girder/slicer_cli_web)

To build a Docker Image from this CLI Plugin,

First pull Nuclei-Detection TensorFlow pre-trained model files and place them inside the 'cli' folder because these files
are going to be placed inside the Docker Image.

wget <link>


then run,

```
$ docker build -t <DockerImage_name>:<DockerImage_tag_version> <Dockerfile_path>
```

To check the Docker Image is completely running,

1. First create a Docker Container of this Docker Image and navigate into /bin/bash
```
$ docker run -ti -v <local_volume_folder>:<Docker_volume_folder> --rm --entrypoint=/bin/bash <DockerImage_name>:<DockerImage_tag_version>
```
  Note : -v : used for mouting local and Docker folders so that the changes to the folder will be mirrored immediately.

2. Your default working directory inside the Container bash is '/Applications/', so run a sample test run.
```
$ python FasterNuclieDetection/FasterNuclieDetection.py ../91316_leica_at2_40x.svs.38552.50251.624.488.jpg annot.anot timeprofile.csv
```
3. Check the annotation and timeprofile files in the local folder.





