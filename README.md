# CNNCellDetection
This repository contains adaptations of the state-of-the-art deep learning based object detection models for the detection of cell nuclei in histology images




First you have to clone the Luminoth repo either from its official repo or the forked repo, which has some modifications made internally.

1. Official repo: https://github.com/tryolabs/luminoth.git
2. Forked Modified repo: https://github.com/cramraj8/luminoth.git



# Data Generation


Raw data can be found in different formats. However, we need to bring that to **PascalVoc** format in order to train the model.
PascalVoc data folder will look like


1. Folder structure
----------------

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
    ├── JPEGImages                      - Folder contains JPEG/PNG images.
    │
    └── test_JPEGImages                 - Folder contains test JPEG/PNG images for inference and testing.


```




## 2. Example

If our raw dataset has this following format of annotations, we need to convert the csv file to xml file.

PascalVoc format recommeds annotations in XML.
Basic information we need from csv file are,
```
    1. annotations for each image converted into each XML file
            a. image_id
            b. each bounding box coordinates in either convention
                - x_min, y_min, x_max, y_max
                - x_center, y_center, width, height
            c. class label for each bounding box objects
                for class-agnostic model, we can represent objects into objectness class
```


## 2.1 XML format:


```
    root: annotation: version: 1.0
        subelement: folder: <project_name>
        subelement: filename: <image_name>
        subelement: source
            subelement: database: database_name
            subelement: annotation: <project_name>
            subelement: image: None
            subelement: flickrid: None
        subelement: owner: <author_name>
        subelement: size
            subelement: width : <width_value>
            subelement: height : <height_value>
            subelement: depth : <depth_value>
        subelement: segmented: '0'

    .... continues to include each objects ....

        subelement: object
            subelement: name: <class_name>
            subelement: pose: 'Unspecified'
            subelement: truncated: '0'
            subelement: difficult: '0'
            subelement: bndbox
                subelement: xmin : <xmin_value>
                subelement: ymin : <ymin_value>
                subelement: xmax : <xmax_value>
                subelement: ymax : <ymax_value>

            Above 'object block' repeatedly added to the root to represent each object properties.


```

to create txt files inside **ImageSets > Main** , by running the **write_txt.py**



## To create tfrecord data

Place **image**, **annotations**, **train.txt** in appropriate arrangements inside the '**pascalvoc_format_data**' folder.

* if you are using Mac, make sure to delete all **.DS_Store** files in each folder hierarchies inside '**pascalvoc_format_data.**')


CLI:
```
    $ lumi dataset transform \
        --type pascal \
        --data-dir ./data/pascalvoc_format_data \
        --output-dir ./data/tf_dataset \
        --split train
```

you may want to execute this command by standing inside **luminoth/** folder.








# Training
    1. place tf_dataset inside ./data/tfrecord/
    2. install luminoth while inside ./luminoth/ subfolder
        $ pip install -e .
        make sure you installed tesorflow-gpu (verified version tensorflow-gpu==1.5.0)
    3. to compose tf_dataset,
        $ lumi dataset transform \
        --type pascal \
        --data-dir ./data/LuminothStyle_Data \
        --output-dir tf_dataset \
        --split train
    4. to train the model using tf_dataset
        $ lumi train -c examples/sample_config.yml




# Evaluation

Two methods of evaluation conducted,

1. Regular method followed by Pascal, COCO challenge, and Luminoth
2. Method used in WBCProject using Hungarian Algorithm for mapping GT with Predictions
