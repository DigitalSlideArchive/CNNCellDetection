# -*- coding: utf-8 -*-
# @__ramraj__

"""
This code will only work if the luminoth model was edited to include the tf-variable-name assignments using the
API tf.identity(<variable>, '<variable_name_we_want_to_call>')

This is basically like we insert some reference_names to some variables inside the model graph so that at inference stage
we can invoke them to get the result by feeding some input.
"""


import tensorflow as tf
import cv2
import numpy as np
import vis  # This is from luminoth/luminoth/vis.py


CKPT_INDEX = 36000
META_DIR = "./jobs/my-run/model.ckpt-%s.meta" % CKPT_INDEX
CKPT_DIR = "./jobs/my-run/model.ckpt-%s" % CKPT_INDEX

N_MAX_DET = 800
MIN_PROB = 0.1


def restore_meta_ckpt():
    img = cv2.imread('img1.png')
    img = np.array(img)
    print(img.shape)

    saver = tf.train.import_meta_graph(META_DIR)

    imported_graph = tf.get_default_graph()

    # Saving Graph property names

    # graph_op = imported_graph.get_operations()
    # with open('TF_output_operations.txt', 'w') as f:
    #     for i in graph_op:
    #         f.write(str(i))
    graph_vars = tf.get_default_graph().get_collection("variables")
    with open('TF_output_vars.txt', 'w') as f:
        for i in graph_vars:
            f.write(str(i))
    # graph_collec = tf.get_default_graph().get_all_collection_keys()
    # with open('TF_output_collections.txt', 'w') as f:
    #     for i in graph_collec:
    #         f.write(str(i))
    graph_nodes = tf.get_default_graph().as_graph_def().node
    with open('TF_output_nodes.txt', 'w') as f:
        for i in graph_nodes:
            f.write(str(i))

    graph_x = tf.get_default_graph().get_tensor_by_name(
        'fasterrcnn/input_image_tensor:0')
    print(graph_x)

    graph_y_objects = tf.get_default_graph().\
        get_tensor_by_name('fasterrcnn/output_objects:0')
    print(graph_y_objects)
    graph_y_labels = tf.get_default_graph().\
        get_tensor_by_name('fasterrcnn/output_labels:0')
    print(graph_y_labels)
    graph_y_probs = tf.get_default_graph().\
        get_tensor_by_name('fasterrcnn/output_probs:0')
    print(graph_y_objects)
    print(graph_y_labels)
    print(graph_y_probs)

    class_max_detect_input = tf.get_default_graph().get_tensor_by_name(
        'fasterrcnn/class_max_detect_input:0')
    total_max_detect_input = tf.get_default_graph().get_tensor_by_name(
        'fasterrcnn/total_max_detect_input:0')
    min_prob_input = tf.get_default_graph().get_tensor_by_name(
        'fasterrcnn/min_prob_threshold_input:0')

    print(class_max_detect_input)
    print(total_max_detect_input)
    print(min_prob_input)

    with tf.Session() as sess:
        saver.restore(sess, CKPT_DIR)

        # init = tf.group(tf.global_variables_initializer(),
        #                 tf.local_variables_initializer())
        init = tf.group(tf.local_variables_initializer())
        sess.run(init)

        coord_graph = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord_graph)

        # objects_val, labels_val, probs_val = sess.run([graph_y_objects,
        #                                                graph_y_labels,
        #                                                graph_y_probs],
        #                                               feed_dict={graph_x: img})
        # print(objects_val)
        # print(labels_val)
        # print(probs_val)

        objects_tf = graph_y_objects
        objects_labels_tf = graph_y_labels
        objects_labels_prob_tf = graph_y_probs

        fetches = {
            'objects': objects_tf,
            'labels': objects_labels_tf,
            'probs': objects_labels_prob_tf,
        }

        fetched = sess.run(fetches, feed_dict={graph_x: img,
                                               class_max_detect_input: N_MAX_DET,
                                               total_max_detect_input: N_MAX_DET,
                                               min_prob_input: MIN_PROB})

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

        # with open('zzzzzzzZZZZ', 'w') as f:
        #     for i in predictions:
        #         f.write(str(i))

        with open('fullyloadbale_predict_calling_outputIMG.txt', 'w') as f:
            f.write(str(predictions))

        # with open('PRED_objects', 'w') as f:
        #     for i in objects_val:
        #         f.write(str(i))

        # with open('PRED_labels.txt', 'w') as f:
        #     for i in labels_val:
        #         f.write(str(i))

        # with open('PRED_probs.txt', 'w') as f:
        #     for i in probs_val:
        #         f.write(str(i))

        # Finish off the filename queue coordinator.
        coord_graph.request_stop()
        coord_graph.join(threads)

        pred_img = vis.vis_objects(img, predictions)
        cv2.imwrite('fullyloadbale_predict_calling_outputIMG.png',
                    np.asarray(pred_img))


restore_meta_ckpt()
