# -*- coding: utf-8 -*-
# @__ramraj__


from luminoth.utils.config import get_config
from luminoth.utils.predicting import PredictorNetwork

from luminoth.models import get_model
from luminoth.datasets import get_dataset

import tensorflow as tf
import cv2
import numpy as np
import os


CONFIG = './examples/sample_config.yml'
MAX_DET = 800
MIN_PROB = 0.5

config = get_config(CONFIG)
config.model.rcnn.proposals.total_max_detections = MAX_DET
config.model.rcnn.proposals.min_prob_threshold = MIN_PROB


CKPT_INDEX = 36000
META_DIR = "./jobs/my-run/model.ckpt-%s.meta" % CKPT_INDEX
CKPT_DIR = "./jobs/my-run/model.ckpt-%s" % CKPT_INDEX


# def method_outside_calling():  # WORKING
#     img = cv2.imread('delete.png', 0)
#     img = np.expand_dims(img, -1)
#     network = PredictorNetwork(config)
#     objects = network.predict_image(img)


def method_inside_graph_calling(file):  # WORKING

    print('File Name : ************** : ', file)

    INPUT_IMG = './testJPEG/CroppedROI/' + file
    OUTPUT_IMG = './' + file
    OUTPUT_TXT = '%s.txt' % OUTPUT_IMG[:-4]

    img = cv2.imread(INPUT_IMG)
    # img = np.expand_dims(img, -1)

    dataset_class = get_dataset('object_detection')
    model_class = get_model('fasterrcnn')
    dataset = dataset_class(config)
    model = model_class(config)

    graph = tf.Graph()
    session = tf.Session(graph=graph)

    with graph.as_default():
        image_placeholder = tf.placeholder(
            tf.float32, (None, None, 3), name='Input_Placeholder'
        )
        pred_dict = model(image_placeholder)

        # Restore checkpoint
        job_dir = META_DIR
        # ckpt = tf.train.get_checkpoint_state(job_dir)
        # ckpt = ckpt.all_model_checkpoint_paths[-1]
        # saver = tf.train.Saver(sharded=True, allow_empty=True)
        # saver.restore(session, ckpt)

        saver = tf.train.Saver(sharded=True, allow_empty=True)
        saver.restore(session, CKPT_DIR)
        tf.logging.info('Loaded checkpoint.')

        cls_prediction = pred_dict['classification_prediction']
        objects_tf = cls_prediction['objects']
        objects_labels_tf = cls_prediction['labels']
        objects_labels_prob_tf = cls_prediction['probs']

        fetches = {
            'objects': objects_tf,
            'labels': objects_labels_tf,
            'probs': objects_labels_prob_tf,
        }

        fetched = session.run(fetches, feed_dict={
            image_placeholder: np.array(img)
        })

        objects = fetched['objects']
        labels = fetched['labels'].tolist()
        probs = fetched['probs'].tolist()

        # Cast to int to consistently return the same type in Python 2 and 3
        objects = [
            [int(round(coord)) for coord in obj]
            for obj in objects.tolist()
        ]

        predictions = sorted([
            {
                'bbox': obj,
                'label': label,
                'prob': round(prob, 4),
            } for obj, label, prob in zip(objects, labels, probs)
        ], key=lambda x: x['prob'], reverse=True)

        with open(OUTPUT_TXT, 'w') as f:
            for i in predictions:
                f.write(str(i))

        import vis

        pred_img = vis.vis_objects(img, predictions)
        cv2.imwrite(OUTPUT_IMG, np.asarray(pred_img))


if __name__ == '__main__':

    FILES = os.listdir('./testJPEG/CroppedROI/')
    print FILES

    method_inside_graph_calling(FILES[3])

    # for file in FILES:
    #     method_inside_graph_calling(file)
