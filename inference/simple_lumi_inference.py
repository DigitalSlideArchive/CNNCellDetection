# -*- coding: utf-8 -*-
# @__ramraj__


from luminoth.utils.config import get_config
from luminoth.utils.predicting import PredictorNetwork
import tensorflow as tf

CONFIG = './lumi_det/examples/sample_config.yml'
MAX_DET = 1000
MIN_PROB = 0.1

config = get_config(CONFIG)
config.model.rcnn.proposals.total_max_detections = MAX_DET
config.model.rcnn.proposals.min_prob_threshold = MIN_PROB

network = PredictorNetwork(config, graph)
objects = network.predict_image(input_im)
