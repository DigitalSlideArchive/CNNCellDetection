# -*- coding: utf-8 -*-
# @__ramraj__


# Nuclei Detection with Dask


from __future__ import absolute_import, division, print_function


from luminoth.utils.config import get_config
from luminoth.utils.predicting import PredictorNetwork
from luminoth.models import get_model
from luminoth.datasets import get_dataset


import os
import sys
import time
import copy
import logging
import itertools
import large_image
import numpy as np
import scipy as sp
import collections
import pandas as pd
import ujson as json


import dask
import dask.distributed

import tensorflow as tf
import utils as cli_utils


import histomicstk as htk
import histomicstk.preprocessing.color_normalization as htk_cnorm
import histomicstk.preprocessing.color_deconvolution as htk_cdeconv
import histomicstk.features as htk_features
import histomicstk.utils as htk_utils
import histomicstk.segmentation.nuclear as htk_nuclear
import histomicstk.segmentation.label as htk_seg_label
import histomicstk.filters.shape as htk_shape_filters


from skimage.filters import threshold_yen, threshold_otsu, threshold_isodata


logging.basicConfig(level=logging.CRITICAL)


CONFIG = '/home/cramraj8/Kitware/NuclieDetection-HistomicsTK-Dask-Plugin/lumi/examples/sample_config.yml'


CKPT_INDEX = 36000
JOB_DIR = 'jobs'
META_DIR = "/home/cramraj8/Kitware/NuclieDetection-HistomicsTK-Dask-Plugin/lumi/%s/my-run/model.ckpt-%s.meta" % \
    (JOB_DIR, CKPT_INDEX)
CKPT_DIR = "/home/cramraj8/Kitware/NuclieDetection-HistomicsTK-Dask-Plugin/lumi/%s/my-run/model.ckpt-%s" % \
    (JOB_DIR, CKPT_INDEX)


args = {


    # ========== Image 1 ==========
    # 'inputImageFile': '/home/cramraj8/Kitware/101625.svs',
    'inputImageFile': '/home/cramraj8/Kitware/0014_BCL2_02.svs',
    # 'analysis_roi': [7540, 6700, 2400, 2400],
    'analysis_roi': [-1, -1, -1, -1],
    'outputTimeProfilingFile': '/home/cramraj8/Kitware/NuclieDetection-HistomicsTK-Dask-Plugin/CLUSTER----Timeprofiling---CPU---jobs_resnet50_ff1ep2---whole_WSI---n_samples40---101625.csv',
    'output_annotation_dir': '/home/cramraj8/Kitware/NuclieDetection-HistomicsTK-Dask-Plugin/',


    'max_det': 1000,
    'min_prob': 0.1,

    'reference_mu_lab': [8.97307880463709, -0.048069533099968385, -0.007750513198518623],
    'reference_std_lab': [0.35412366, 0.08349332, 0.01101242],

    'stain_1': 'hematoxylin',
    'stain_2': 'dab',
    'stain_3': 'null',
    'stain_1_vector': [-1, -1, -1],
    'stain_2_vector': [-1, -1, -1],
    'stain_3_vector': [-1, -1, -1],
    'deconv_method': 'ruifrok',

    'ignore_border_nuclei': True,

    'nuclei_annotation_format': 'bbox',  # Must be 'bbox' or 'boundary'

    'morphometry_features': False,
    'fsd_features': False,
    'intensity_features': True,
    'gradient_features': True,
    'haralick_features': True,
    'cytoplasm_features': False,
    'cyto_width': 5,
    'fsd_bnd_pts': 128,
    'fsd_freq_bins': 6,
    'num_glcm_levels': 32,
    'feature_file_format': 'hdf',

    'min_fgnd_frac': 0.75,
    'analysis_mag': 40,
    'analysis_tile_size': 1024,


    'scheduler': 'tcp://hpg6-112.maas:8786',
    'num_workers': -1,
    'num_threads_per_worker': 1
}

args = collections.namedtuple('Parameters', args.keys())(**args)


def detect_tile_nuclei(slide_path, tile_position, args, it_kwargs,
                       src_mu_lab=None, src_sigma_lab=None, debug=False):

    # =========================================================================
    # ======================= Tile Loading ====================================
    # =========================================================================
    print('\n>> Loading Tile ... \n')

    csv_dict = {}

    csv_dict['PreparationTime'] = []
    csv_dict['ColorDeconvTime'] = []
    csv_dict['TotalTileLoadingTime'] = []

    csv_dict['CKPTLoadingTime'] = []
    csv_dict['ModelInfernceTime'] = []
    csv_dict['DetectionTime'] = []

    csv_dict['ROIShape'] = []
    csv_dict['ObjectsDict'] = []
    csv_dict['NumObjects'] = []

    csv_dict['AnnotationWritingTime'] = []

    csv_dict['AnnotationDict'] = []

    start_time = time.time()
    total_tileloading_start_time = time.time()

    ts = large_image.getTileSource(slide_path)
    tile_info = ts.getSingleTile(
        tile_position=tile_position,
        format=large_image.tilesource.TILE_FORMAT_NUMPY,
        **it_kwargs)
    im_tile = tile_info['tile'][:, :, :3]
    csv_dict['ROIShape'] = im_tile.shape[:2]

    prep_time = time.time() - start_time
    csv_dict['PreparationTime'] = round(prep_time, 3)

    # =========================================================================
    # =================Img Normalization & Color Deconv========================
    # =========================================================================
    print('\n>> Color Deconvolving ... \n')
    start_time = time.time()

    im_nmzd = htk_cnorm.reinhard(
        im_tile,
        args.reference_mu_lab,
        args.reference_std_lab,
        src_mu=src_mu_lab,
        src_sigma=src_sigma_lab
    )

    # perform color decovolution
    if args.deconv_method == 'ruifrok':

        w = cli_utils.get_stain_matrix(args)
        im_stains = htk_cdeconv.color_deconvolution(
            im_nmzd, w).Stains.astype(np.float)[:, :, :2]

    elif args.deconv_method == 'macenko':

        w_est = htk_cdeconv.rgb_separate_stains_macenko_pca(im_tile, 255)
        im_stains = htk_cdeconv.color_deconvolution(
            im_tile, w_est, 255).Stains.astype(np.float)
        ch1 = htk_cdeconv.find_stain_index(
            htk_cdeconv.stain_color_map[args.stain_1], w_est)
        ch2 = htk_cdeconv.find_stain_index(
            htk_cdeconv.stain_color_map[args.stain_2], w_est)
        im_stains = im_stains[:, :, [ch1, ch2]]

    else:

        raise ValueError('Invalid deconvolution method parameter.')

    # =========================================================================
    # ====================== Fuse the stain1 & stain2 pix======================
    # =========================================================================

    # compute nuclear foreground mask
    im_fgnd_mask_stain_1 = im_stains[
        :, :, 0] < threshold_yen(im_stains[:, :, 0])
    im_fgnd_mask_stain_2 = im_stains[
        :, :, 1] < threshold_yen(im_stains[:, :, 1])
    im_fgnd_seg_mask = im_fgnd_mask_stain_1 | im_fgnd_mask_stain_2

    # segment nuclei
    im_nuc_det_input = np.squeeze(np.min(im_stains[:, :, :2], axis=2))
    print('---> Fusing 2 Stains')
    deconv_time = time.time() - start_time
    csv_dict['ColorDeconvTime'] = round(deconv_time, 3)

    # =========================================================================
    # ================= Nuclie Detection Deep Learning Block ==================
    # =========================================================================

    total_tileloading_time = time.time() - total_tileloading_start_time
    csv_dict['TotalTileLoadingTime'] = round(total_tileloading_time, 3)

    start_time = time.time()

    config = get_config(CONFIG)
    config.model.rcnn.proposals.total_max_detections = args.max_det
    config.model.rcnn.proposals.min_prob_threshold = args.min_prob
    im_nuc_det_input = np.stack((im_nuc_det_input,) * 3, axis=-1)

    # ====================================================================================================================================
    tf.reset_default_graph()

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

        ckpt_loading_start_time = time.time()

        saver = tf.train.Saver(sharded=True, allow_empty=True)
        saver.restore(session, CKPT_DIR)
        tf.logging.info('Loaded checkpoint.')

        ckpt_loading_time = time.time() - ckpt_loading_start_time
        csv_dict['CKPTLoadingTime'] = round(ckpt_loading_time, 3)

        inference_start_time = time.time()

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
            image_placeholder: np.array(im_nuc_det_input)
        })

        inference_time = time.time() - inference_start_time
        csv_dict['ModelInfernceTime'] = round(inference_time, 3)

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

    print('\n>> Finishing Detection ... \n')
    print('***** Number of Detected Cells ****** : ', len(predictions))
    detection_time = time.time() - start_time
    csv_dict['DetectionTime'] = round(detection_time, 3)
    csv_dict['NumObjects'] = len(predictions)
    csv_dict['ObjectsDict'] = predictions

    # =========================================================================
    # ======================= TODO: Implement border deletion =================
    # =========================================================================

    # =========================================================================
    # ======================= Write Annotations ===============================
    # =========================================================================

    start_time = time.time()

    objects_df = pd.DataFrame(objects)
    formatted_nuclei_det_list = cli_utils.convert_preds_to_utilformat(
        objects_df,
        probs,
        args.ignore_border_nuclei,
        im_tile_size=args.analysis_tile_size)

    nuclei_annot_list = cli_utils.create_tile_nuclei_annotations(
        formatted_nuclei_det_list, tile_info, args.nuclei_annotation_format)
    csv_dict['AnnotationDict'] = nuclei_annot_list

    num_nuclei = len(nuclei_annot_list)

    anot_time = time.time() - start_time
    csv_dict['AnnotationWritingTime'] = round(anot_time, 3)

    # =========================================================================
    # ======================= Compute Features ================================
    # =========================================================================
    """
    # compute nuclei features
    start_time = time.time()

    stain_names = [args.stain_1, args.stain_2]

    fdata = [None] * 2

    for i in range(2):

        fdata[i] = htk_features.compute_nuclei_features(
            im_nuclei_seg_mask, im_stains[:, :, i],
            fsd_bnd_pts=args.fsd_bnd_pts,
            fsd_freq_bins=args.fsd_freq_bins,
            num_glcm_levels=args.num_glcm_levels,
            morphometry_features_flag=(args.morphometry_features and i == 0),
            fsd_features_flag=(args.fsd_features and i == 0),
            intensity_features_flag=args.intensity_features,
            gradient_features_flag=args.gradient_features
        )

        fdata[i].columns = ['Feature.{}.'.format(
            stain_names[i]) + col for col in fdata[i].columns]

    fdata = pd.concat(fdata, axis=1)

    fex_time = time.time() - start_time

    slide_name = os.path.basename(slide_path)
    print('{}, Tile {}, Num nuclei = {}, load_time = {}, nuc_time = {}, anot_time = {}, fex_time = {}'.format(
        slide_name, tile_position, num_nuclei, prep_time, nuc_time, anot_time, fex_time))

    assert len(nuclei_annot_list) == len(
        fdata), 'ERROR - {}, Tile {}, num_nuclei != num_feature_rows'.format(slide_name, tile_position)

    return nuclei_annot_list, fdata, (prep_time, anot_time, nuc_time, fex_time)
    """

    return csv_dict


def save_nuclei_annotation(nuclei_anot_list, anot_name, out_file_name):

    if isinstance(anot_name, list):

        assert(len(anot_name) == nuclei_anot_list)

        annotation = [
            {
                "name": anot_name[i],
                "elements": nuclei_anot_list[i]
            }

            for i in range(len(anot_name))
        ]

    else:

        annotation = {
            "name":     anot_name,
            "elements": nuclei_anot_list
        }

    out_file = out_file_name + '.anot'

    with open(out_file, 'w') as annotation_file:
        json.dump(annotation, annotation_file)


def main(args):

    total_time_profiler = {}

    total_start_time = time.time()

    # =========================================================================
    # ======================= Create Dask Client ==============================
    # =========================================================================
    print('\n>> Creating Dask client ...\n')

    start_time = time.time()
    c = cli_utils.create_dask_client(args)
    print(c)
    dask_setup_time = time.time() - start_time
    temp_time = cli_utils.disp_time_hms(dask_setup_time)
    print('Dask setup time = {}'.format(
        temp_time))
    total_time_profiler['Dask setup time'] = temp_time

    # =========================================================================
    # ========================= Read Input Image ==============================
    # =========================================================================

    print('\n>> Reading input image ... \n')

    ts = large_image.getTileSource(args.inputImageFile)
    ts_metadata = ts.getMetadata()

    print(json.dumps(ts_metadata, indent=2))
    if np.all(np.array(args.analysis_roi) == -1):
        process_whole_image = True
    else:
        process_whole_image = False
    is_wsi = ts_metadata['magnification'] is not None

    # =========================================================================
    # ===================== Compute Foreground Mask ===========================
    # =========================================================================

    if is_wsi and process_whole_image:

        print('\n>> Computing tissue/foreground mask at low-res ...\n')

        start_time = time.time()

        im_fgnd_mask_lres, fgnd_seg_scale = \
            cli_utils.segment_wsi_foreground_at_low_res(ts)

        fgnd_time = time.time() - start_time

        tmp_time = cli_utils.disp_time_hms(fgnd_time)
        print('low-res foreground mask computation time = {}'.format(tmp_time))
        total_time_profiler[
            'low-res foreground mask computation time'] = tmp_time

    # =========================================================================
    # ================== Compute foreground fraction ==========================
    # =========================================================================
    it_kwargs = {
        'tile_size': {'width': args.analysis_tile_size},
        'scale': {'magnification': args.analysis_mag},
        'resample': True
    }
    tile_fgnd_frac_list = [1.0]
    if not process_whole_image:

        it_kwargs['region'] = {
            'left': args.analysis_roi[0],
            'top': args.analysis_roi[1],
            'width': args.analysis_roi[2],
            'height': args.analysis_roi[3],
            'units': 'base_pixels'
        }
    # =========================================================================
    if is_wsi:
        print('\n>> Computing foreground fraction of all tiles ...\n')

        start_time = time.time()

        num_tiles = ts.getSingleTile(**it_kwargs)['iterator_range']['position']

        print('Number of tiles = {}'.format(num_tiles))

        if process_whole_image:
            tile_fgnd_frac_list = htk_utils.compute_tile_foreground_fraction(
                args.inputImageFile, im_fgnd_mask_lres, fgnd_seg_scale,
                it_kwargs
            )

        else:

            tile_fgnd_frac_list = np.full(num_tiles, 1.0)

        num_fgnd_tiles = np.count_nonzero(
            tile_fgnd_frac_list >= args.min_fgnd_frac)

        percent_fgnd_tiles = 100.0 * num_fgnd_tiles / num_tiles

        fgnd_frac_comp_time = time.time() - start_time

        print('Number of foreground tiles = {:d} ({:2f}%)'.format(
            num_fgnd_tiles, percent_fgnd_tiles))

        print('Tile foreground fraction computation time = {}'.format(
            cli_utils.disp_time_hms(fgnd_frac_comp_time)))

    # =========================================================================
    # ========================= Compute reinhard stats ========================
    # =========================================================================
    src_mu_lab = None
    src_sigma_lab = None

    print('\n>> Computing reinhard color normalization stats ...\n')

    start_time = time.time()

    src_mu_lab, src_sigma_lab = htk_cnorm.reinhard_stats(
        args.inputImageFile, 0.01, magnification=args.analysis_mag,
        tissue_seg_mag=0.625)

    print('Reinahrd stats')
    print(src_mu_lab, src_sigma_lab)

    rstats_time = time.time() - start_time

    print('Reinhard stats computation time = {}'.format(
        cli_utils.disp_time_hms(rstats_time)))

    # =========================================================================
    # ======================== Detect Nuclie in Parallel -  Dask ==============
    # =========================================================================
    print('\n>> Detecting cell ...\n')
    start_time = time.time()

    prep_time_profiler = []
    color_deconv_time_profiler = []
    total_loading_time_profiler = []
    ckpt_loading_time_profiler = []
    model_inference_time_profiler = []
    detection_time_profiler = []
    tile_shapes = []
    tile_nuclei_list = []
    num_nuclie = []
    annotation_dict = []
    annotation_dict_list = []
    nuclei_annot_list = []

    try:
        # count = 0
        for tile in ts.tileIterator(**it_kwargs):
            # if count > 40:
            #     break
            # else:
            #     count = count + 1

            tile_position = tile['tile_position']['position']
            if is_wsi and tile_fgnd_frac_list[tile_position] <= args.min_fgnd_frac:
                continue
            if is_wsi and process_whole_image and (tile['width'] != args.analysis_tile_size or tile['height'] != args.analysis_tile_size):
                continue

            tmp_csv = dask.delayed(detect_tile_nuclei)(
                args.inputImageFile,
                tile_position,
                args, it_kwargs,
                src_mu_lab, src_sigma_lab
            )

            prep_time_profiler.append(tmp_csv['PreparationTime'])
            color_deconv_time_profiler.append(tmp_csv['ColorDeconvTime'])
            total_loading_time_profiler.append(tmp_csv['TotalTileLoadingTime'])
            ckpt_loading_time_profiler.append(tmp_csv['CKPTLoadingTime'])
            model_inference_time_profiler.append(tmp_csv['ModelInfernceTime'])
            detection_time_profiler.append(tmp_csv['DetectionTime'])
            tile_shapes.append(tmp_csv['ROIShape'])
            tile_nuclei_list.append(tmp_csv['ObjectsDict'])
            num_nuclie.append(tmp_csv['NumObjects'])
            annotation_dict.append(tmp_csv['AnnotationDict'])

        prep_time_profiler,\
            color_deconv_time_profiler,\
            total_loading_time_profiler,\
            ckpt_loading_time_profiler,\
            model_inference_time_profiler,\
            detection_time_profiler,\
            tile_shapes,\
            tile_nuclei_list,\
            num_nuclie,\
            annotation_dict = dask.compute(prep_time_profiler,
                                           color_deconv_time_profiler,
                                           total_loading_time_profiler,
                                           ckpt_loading_time_profiler,
                                           model_inference_time_profiler,
                                           detection_time_profiler,
                                           tile_shapes,
                                           tile_nuclei_list,
                                           num_nuclie,
                                           annotation_dict
                                           )

        nuclei_annot_list = list(
            itertools.chain.from_iterable(list(tile_nuclei_list)))

        num_nuclei = len(nuclei_annot_list)

        nuclei_detection_time = time.time() - start_time

        print('Number of nuclei = {}'.format(num_nuclei))
        print('Nuclei detection time = {}'.format(
            cli_utils.disp_time_hms(nuclei_detection_time)))

        annotation_dict_list = list(
            itertools.chain.from_iterable(list(annotation_dict)))

    finally:
        agg_csv = {}
        agg_csv['PreparationTime'] = prep_time_profiler
        agg_csv['ColorDeconvTime'] = color_deconv_time_profiler
        agg_csv['TotalTileLoadingTime'] = total_loading_time_profiler
        agg_csv['CKPTLoadingTime'] = ckpt_loading_time_profiler
        agg_csv['ModelInfernceTime'] = model_inference_time_profiler
        agg_csv['DetectionTime'] = detection_time_profiler
        agg_csv['ROIShape'] = tile_shapes
        agg_csv['ObjectsDict'] = tile_nuclei_list
        agg_csv['NumObjects'] = num_nuclie

        df = pd.DataFrame(agg_csv,
                          columns=['PreparationTime', 'ColorDeconvTime',
                                   'TotalTileLoadingTime',
                                   'CKPTLoadingTime', 'ModelInfernceTime',
                                   'DetectionTime',
                                   'ROIShape',
                                   # 'ObjectsDict',
                                   'NumObjects']
                          )
        df.to_csv(args.outputTimeProfilingFile)

        df = pd.DataFrame(annotation_dict_list)
        df.to_csv('Annotations.csv')
        #
        # Write annotation file
        #
        print('\n>> Writing annotation file ...\n')

        out_file_name = os.path.basename(args.inputImageFile)
        out_file_prefix = os.path.join(
            args.output_annotation_dir, out_file_name)
        annot_name = out_file_name + '-whole-slide-nuclei-fasterRCNN'

        save_nuclei_annotation(nuclei_annot_list, annot_name, out_file_prefix)

    # ====================================================================================
    # ====================================================================================
    # ====================================================================================


if __name__ == "__main__":

    main(args)
