# CNNCellDetection
This repository contains adaptations of the state-of-the-art deep learning based object detection models for the detection of cell nuclei in histology images




First you have to clone the Luminoth repo either from its official repo or the forked repo, which has some modifications made internally.

1. Official repo: https://github.com/tryolabs/luminoth.git
2. Forked Modified repo: https://github.com/cramraj8/luminoth.git

Forked repo has,

* ResNet block 3 **num_units** hyper-parameter exposed in base_config.yml for all ResNet variants
* provided ```img = input_img[:, :, :num_channels]``` in dataset loading function
 to facilitate gray image loading and unncessary TensorFlow reshaping exceptions.
* provided end-points in feature extraction from R-CNN layers
* exposed base_config.yml in the root folder of the application



# Data Generation


Raw data can be found in different formats. Either in csv file or **PascalVoc** format in order to train the model.

1. csv file only needs to contain the following columns, and the columns names can be overidden by input argument.
```
    a. image_id
    b. bounding box coordinates in either convention
            - x_min, y_min, x_max, y_max
            - x_center, y_center, width, height
    c. class label(for class-agnostic model, represent by objectness class)
```

2. PascalVoc data folder should look like below,



```
   Data
    ├── annotations                     - Folder contains XML ground truth annotations.
    │
    │
    ├── ImageSets
    │   └──Main
    │       ├── objectness_train.txt    - contains the image_ids that has this particular 'objectness' class.
    │       └── train.py                - contains the image_ids that is going to be used for training.
    │
    │
    └── JPEGImages                      - Folder contains JPEG/PNG images.
```





## To create tfrecord data

Either place **image**, **annotations**, **train.txt** in appropriate arrangements inside the '**pascalvoc_format_data**' folder or
have a csv file in appropriate format.



## Luminoth CLI:

from PascalVoc format to tfrecord generation
```
    $ lumi dataset transform \
        --type pascal \
        --data-dir ./data/pascalvoc_format_data \
        --output-dir ./data/tf_dataset \
        --split train
```


from csv format to tfrecord generation
```
    $ lumi dataset transform \
        --type csv \
        --data-dir ./data/csv_file \
        --output-dir ./data/tf_dataset \
        --split train
```


you may want to execute this command by standing inside **luminoth/** folder.








# Training
```
$ lumi train -c train/sample_config.yml
```




# Evaluation

Two methods of evaluation conducted,

1. IoU based mAP evaluation method
2. Objectness/classification confidence score based AUC evaluation method
        using Hungarian Algorithm for mapping GT with Predictions

Generally, in the PascalVOC and COCO object detection challenges people often use method #1; however, this method
is better fit for the problem of detecting small number of objects per image. But in the nuclei-detection problem domain,
we usually face more than 100 nuclei(objects) in each image. In this case, without mapping best prediction bndbox with ground-truth,
it is hard to identify the redundant detection bndboxes based on method #1.

The example of evaluation results overlayed over the input image using method #2 because of the main reason - crowded bndbox detections.
Green boxes are TPs, blue boxes are FP, and red boxes are FNs respectively.

<!-- ![alt text](https://github.com/DigitalSlideArchive/CNNCellDetection/evaluation/ex1-overlay_TCGA-G9-6362-01Z-00-DX1_3.png) -->
<!-- ![Alt text](evaluation/ex1-overlay_TCGA-G9-6362-01Z-00-DX1_3.png=250x250?raw=true "Title") -->

<!-- ![test image size](evaluation/ex1-overlay_TCGA-G9-6362-01Z-00-DX1_3.png){:height="45%" width="44%"}
![test image size](evaluation/ex2-overlay_TCGA-HE-7130-01Z-00-DX1_2.png){:height="45%" width="49%"} -->

<!-- ![alt-text-1](evaluation/ex1-overlay_TCGA-G9-6362-01Z-00-DX1_3.png "title-1") ![alt-text-2](evaluation/ex2-overlay_TCGA-HE-7130-01Z-00-DX1_2.png "title-2") -->

Evaluation overlays of prediction example-1                |  Evaluation overlays of prediction example-2
:-------------------------:|:-------------------------:
![](evaluation/ex1-overlay_TCGA-G9-6362-01Z-00-DX1_3.png)  |  ![](evaluation/ex2-overlay_TCGA-HE-7130-01Z-00-DX1_2.png)





# Nuclei Detection Web CLI Plugin

An extended plugin for [girder/slicer_cli_web](https://github.com/girder/slicer_cli_web)

To build a Docker Image from this CLI Plugin,

First pull Nuclei-Detection TensorFlow pre-trained model files and place them inside the 'cli' folder because these files
are going to be placed inside the Docker Image.

*wget <link>*


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





