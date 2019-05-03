# -*- coding: utf-8 -*-
# @__ramraj__


import shutil
import os


TOTAL_XMLs = '../data_preparation/DST_DIR/XML/'

DST_XMLs = './test_XMLs/'
TEST_IMGs = './test_images/'


test_pngs = os.listdir(TEST_IMGs)
all_xmls = os.listdir(TOTAL_XMLs)
for test_png in test_pngs:
    exp_test_xml = '%s.xml' % test_png[:-4]
    for ins_xml in all_xmls:
        if ins_xml == exp_test_xml:
            print 'Found : ', ins_xml

            src_xml_path = TOTAL_XMLs + exp_test_xml
            dst_xml_path = DST_XMLs + ins_xml

            shutil.copy(src_xml_path, dst_xml_path)
