#!/usr/bin/env python
"""
generate_baselines.py creates tsv files containing features for training 
the 'ResNet' baseline from the paper.
"""
from __future__ import print_function
from __future__ import division
import base64
import numpy as np
import cv2
import csv
import json
import os
import caffe
import sys
from scipy.ndimage import zoom
import random
import argparse
random.seed(1)

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
KARPATHY_SPLITS = '/home/shai/Up-Down-Captioner/data/coco_splits/karpathy_%s_images.txt' # train,val,test


IMAGE_DIR = 'data/images/'


def load_karpathy_splits(dataset='train'):
  imgIds = set()
  with open(KARPATHY_SPLITS % dataset) as data_file:
    for line in data_file:
      imgIds.add(int(line.split()[-1]))
  return imgIds


def load_image_ids():
  ''' Map image ids to file paths. '''
  id_to_path = {}
  for fname in [#'image_info_test2014.json',
                 'captions_val2014.json', 'captions_train2014.json']:
    with open('/home/shai/Up-Down-Captioner/data/coco/%s' % fname) as f:
      data = json.load(f)
      for item in data['images']:
        image_id = int(item['id'])
        filepath = item['file_name'].split('_')[1] + '/' + item['file_name']
        id_to_path[image_id] = filepath
  print ('Loaded %d image ids' % len(id_to_path))
  return id_to_path


def get_features_from_prepared(image_id, ft_file,fc7=None, mirrorshift=0):
  ft = np.load(ft_file)
  if fc7 is not None:
      final_ft = np.empty_like(ft)
      final_ft = np.inner(ft,fc7['w'])+fc7['b']
      ft= final_ft
  num_boxes = ft.shape[0]
  prepared_ft = {
      'image_id': image_id+int(mirrorshift),  # this is a mirror file so it requires '1' in the 7th char from the end
      'image_h': 0,
      'image_w': 0,
      'num_boxes': num_boxes,
      'boxes': base64.b64encode(np.zeros((num_boxes,0), dtype=np.float32)),
      'features': base64.b64encode(ft)
  }
  return prepared_ft

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate UpDown input from prepared features',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_features_dir',
                        help='', required=True)
    parser.add_argument('--prefix4tsv',
                        help='', required=True)

    parser.add_argument('--outdir',
                        help='output path',
                        required = True)
    parser.add_argument('--fc7_w',
                        help='')
    parser.add_argument('--fc7_b',
                        help='')
    parser.add_argument('--mirror', default=0,
                        help='', type=int)

    
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
  args = parse_args()
  if not os.path.exists(args.outdir): os.makedirs(args.outdir)
  if not args.mirror:
      tsv_files = ['karpathy_train_{}.tsv'.format(args.prefix4tsv),
              'karpathy_val_{}.tsv'.format(args.prefix4tsv),
              'karpathy_test_{}.tsv'.format(args.prefix4tsv)]
  else:
      tsv_files = ['karpathy_train_{}_mirror.tsv'.format(args.prefix4tsv),
                   'karpathy_val_{}_mirror.tsv'.format(args.prefix4tsv),
                   'karpathy_test_{}_mirror.tsv'.format(args.prefix4tsv)]

  print(tsv_files)  
  train = list(load_karpathy_splits(dataset='train'))
  random.shuffle(train)
  image_id_sets = [train,
               load_karpathy_splits(dataset='val'),
               load_karpathy_splits(dataset='test')]
                
  id_to_path = load_image_ids()
  fc7=None
  if args.fc7_w and args.fc7_b:
      fc7_w = np.load(args.fc7_w)
      fc7_b = np.load(args.fc7_b)
      fc7={"w":fc7_w,"b":fc7_b}
      assert fc7_w.shape[0]==fc7_b.shape[0]
      assert len(fc7_w.shape)==2
  for tsv,image_ids in zip(tsv_files, image_id_sets):
    out_file = os.path.join(args.outdir,tsv)
    with open(out_file, 'wb') as tsv_out:
      print( 'Writing to %s' % out_file)
      writer = csv.DictWriter(tsv_out, delimiter = '\t', fieldnames = FIELDNAMES)
      count = 0
      for image_id in image_ids:
        ft_file = os.path.join(args.input_features_dir,(id_to_path[image_id].split(".jpg")[0]+".npy"))
        prepared_ft = get_features_from_prepared(image_id,ft_file,fc7,1e6*args.mirror)
        writer.writerow(prepared_ft)
        count += 1
        if count % 1000 == 0:
          print ('%d / %d' % (count, len(image_ids)))

