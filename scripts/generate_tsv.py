#!/usr/bin/env python
"""
generate_baselines.py creates tsv files containing features for training 
the 'ResNet' baseline from the paper.
"""

import base64
import numpy as np
import glob
import csv
import json
import os
import caffe
import sys
from scipy.ndimage import zoom
import random
random.seed(1)
from sklearn.decomposition import IncrementalPCA
from sklearn.externals import joblib

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']


def get_dict_from_npys(npys_head, out_file, lst):
  npysfiles = glob.glob(npys_head + '/*.npy')
  print 'number of files in %s = %d' % (dr, len(npysfiles))
  with open(out_file, 'wb') as resnet_tsv_out:
    writer = csv.DictWriter(resnet_tsv_out, delimiter='\t', fieldnames=FIELDNAMES)
    count = 0
    for npyf in npysfiles:
      npyfpath = npyf
      npyf = npyf[:-11] + '1' + npyf[-10:]  # convert it to be mirror format
      if npyf.split('/')[-1].split('.')[0] in lst:
        continue
      npcontent = np.load(npyfpath)
      ResNetDict = {
        'image_id': str(int(npyf.split('/')[-1].split('_')[-1].split('.')[0])),
        'image_h': 1,
        'image_w': 1,
        'num_boxes': len(npcontent),
        'boxes': base64.b64encode(np.zeros((len(npcontent), 0), dtype=np.float32)),
        'features': base64.b64encode(npcontent)
      }

      writer.writerow(ResNetDict)
      count += 1
      if count % 5000 == 0:
        print '%d / %d' % (count, len(npysfiles))
  return

if __name__ == "__main__":
  dir2 = '/media/shai/E/data/pythia/data/detectron/fc6/'
  npys_head = dir2+'vqa_mirror/'

  if not os.path.exists(npys_head+'tsv/'):
    os.makedirs(npys_head+'tsv/')

  test_list_file = '/home/shai/Up-Down-Captioner/data/coco_splits/karpathy_test_images_mirror.txt'
  test_list = []
  with open(test_list_file, 'r') as tlf:
    for lnf in tlf:
      test_list.append(lnf.split(' ')[0].split('/')[-1].split('.')[0])

  dirs_list = []
  for root, dirs, _ in os.walk(npys_head):
    for d in dirs:
      if 'tsv' not in d:      dirs_list.append(os.path.join(root, d))

  # dirs_list = [dirs_list[0]]
  for dr in dirs_list:
    out_file = npys_head+'tsv/' + dr.split('/')[-1] + '.tsv'

    get_dict_from_npys(dr, out_file, test_list)


