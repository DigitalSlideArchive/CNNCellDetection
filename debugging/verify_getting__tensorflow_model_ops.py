# -*- coding: utf-8 -*-
# @__ramraj__


import tensorflow as tf


def get_ops_1_pbtxt():  # difference between pb & pbtxt

    from tensorflow.core.framework import graph_pb2 as gpb
    from google.protobuf import text_format as pbtf

    gdef = gpb.GraphDef()
    with open("/Users/ramA/2019/KWWork/ActualProject/luminoth-master/jobs/my-run/graph.pbtxt", 'r') as fh:
        graph_str = fh.read()
    pbtf.Parse(graph_str, gdef)

    with tf.Graph().as_default() as graph:
        saver = tf.import_graph_def(gdef)

        output_node_names = [
            n.name for n in tf.get_default_graph().as_graph_def().node]
        print output_node_names[:5]

        sess = tf.Session()
        file = tf.train.latest_checkpoint(
            "/Users/ramA/2019/KWWork/ActualProject/luminoth-master/jobs/my-run/")
        print file
        print sess
        print saver
        saver.restore(sess, file)


def get_ops_2_meta():
    # Restore graph to another graph (and make it default graph) and variables
    graph = tf.Graph()

    with graph.as_default():
        saver = tf.train.import_meta_graph(
            "/Users/ramA/2019/KWWork/ActualProject/luminoth-master/jobs/my-run/model.ckpt-1.meta")

        sess = tf.Session()
        op = sess.graph.get_operations()
        print[m.values() for m in op]


def load_meta_convert_proto():

    meta_path = "/Users/ramA/2019/KWWork/ActualProject/luminoth-master/jobs/my-run/model.ckpt-1.meta"  # Your .meta file
    output_node_names = ['output:0']    # Output nodes

    with tf.Session() as sess:

        # Restore the graph
        saver = tf.train.import_meta_graph(meta_path)

        # Load weights
        saver.restore(sess, tf.train.latest_checkpoint(
            "/Users/ramA/2019/KWWork/ActualProject/luminoth-master/jobs/my-run/"))

        # Freeze the graph
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph_def,
            output_node_names)

        # Save the frozen graph
        with open('output_graph.pb', 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())


def restore_meta_ckpt():
    """
    Loaded using meta & ckpt files.
    Called all variables using Collections.
    """
    import tensorflow as tf
    saver = tf.train.import_meta_graph(
        "/Users/ramA/2019/KWWork/ActualProject/luminoth-master/jobs/my-run/model.ckpt-1.meta")
    # print(tf.get_default_graph().get_all_collection_keys())
    # for v in tf.get_default_graph().get_collection("variables"):
    #     print(v)
    # for v in tf.get_default_graph().get_collection("trainable_variables"):
    #     print(v)

    print[n.name for n in tf.get_default_graph().as_graph_def().node]

    imported_graph = tf.get_default_graph()
    graph_op = imported_graph.get_operations()
    with open('output.txt', 'w') as f:
        for i in graph_op:
            f.write(str(i))

    graph_x = tf.get_default_graph().get_tensor_by_name(
        'object_detection_dataset/input_producer/Size:0')

    # input: "object_detection_dataset/TFRecordReaderV2"
    # input: "object_detection_dataset/input_producer"
    # input:
    # "object_detection_dataset/decode_image/cond_jpeg/check_jpeg_channels/x"

    # input:
    # "fasterrcnn/rcnn/rcnn_proposal_1/non_max_suppression/NonMaxSuppressionV2/max_output_size"

    # graph_y = tf.get_default_graph().get_tensor_by_name('layer19/softmax:0')

    sess = tf.Session()
    saver.restore(
        sess, "/Users/ramA/2019/KWWork/ActualProject/luminoth-master/jobs/my-run/model.ckpt-1")
    # result = sess.run("v4:0", feed_dict={"v1:0": 12.0, "v2:0": 4.0})
    # print(result)


if __name__ == '__main__':
    # get_ops_1()  # working
    restore_meta_ckpt()  # working
    # load_meta_convert_proto()
    # get_ops_2_meta()
    # get_ops_1_pbtxt()

    # import numpy as np
    # img = np.random.randint(0, 255, [10, 10])
    # print img.shape


# input_producer
# (<tf.Tensor 'object_detection_dataset/input_producer/Size:0' shape=() dtype=int32>,)
#
