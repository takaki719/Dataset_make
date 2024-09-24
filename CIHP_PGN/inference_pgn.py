from __future__ import print_function
import argparse
from datetime import datetime
import os
import sys
import time
import scipy.misc
import scipy.io as sio
import cv2
from glob import glob
import numpy as np
from PIL import Image
import tensorflow as tf
from utils import *
from utils.model_pgn import PGNModel  # 必要な場合

N_CLASSES = 20
DATA_DIR = './CIHP_PGN/datasets/images'
NUM_STEPS = len(os.listdir(DATA_DIR)) 
print(f"total test images: {NUM_STEPS}")
RESTORE_FROM = './checkpoint/CIHP_pgn'

def load_image(filename):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_image(image_string, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # uint8 を float32 に変換し、正規化
    return image


def main():
    """Create the model and start the evaluation process."""
    # Load dataset
    image_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)]
    dataset = tf.data.Dataset.from_tensor_slices(image_files)
    dataset = dataset.map(load_image)
    dataset = dataset.batch(1)  # バッチサイズを小さく設定する
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    iterator = iter(dataset)
    
    # Create network.
    with tf.compat.v1.variable_scope('', reuse=tf.compat.v1.AUTO_REUSE):
        image_batch = next(iterator)
        image_batch = tf.cast(image_batch, tf.float32)  # uint8 を float32 に変換
        image_rev = tf.reverse(image_batch, axis=[1])
        image_batch = tf.concat([image_batch, image_rev], axis=0)
        net_100 = PGNModel({'data': image_batch}, is_training=False, n_classes=N_CLASSES)
    
    # Create multi-scale images
    h_orig, w_orig = tf.shape(image_batch)[1], tf.shape(image_batch)[2]
    scales = [0.5, 0.75, 1.25, 1.5, 1.75]
    image_batches = [tf.image.resize(image_batch, [tf.cast(h_orig * s, tf.int32), tf.cast(w_orig * s, tf.int32)]) for s in scales]
    
    nets = [PGNModel({'data': b}, is_training=False, n_classes=N_CLASSES) for b in image_batches]

    parsing_outs = [net.layers['parsing_fc'] for net in nets]
    parsing_out2s = [net.layers['parsing_rf_fc'] for net in nets]
    edge_out2s = [net.layers['edge_rf_fc'] for net in nets]

    parsing_out1 = tf.reduce_mean(tf.stack([tf.image.resize(p, tf.shape(image_batch)[1:3]) for p in parsing_outs]), axis=0)
    parsing_out2 = tf.reduce_mean(tf.stack([tf.image.resize(p, tf.shape(image_batch)[1:3]) for p in parsing_out2s]), axis=0)
    edge_out2 = tf.reduce_mean(tf.stack([tf.image.resize(e, tf.shape(image_batch)[1:3]) for e in edge_out2s]), axis=0)

    raw_output = tf.reduce_mean(tf.stack([parsing_out1, parsing_out2]), axis=0)
    head_output, tail_output = tf.unstack(raw_output, num=2, axis=0)
    tail_output_rev = tf.reverse(tail_output, axis=[1])
    
    raw_output_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
    raw_output_all = tf.expand_dims(raw_output_all, axis=0)
    pred_scores = tf.reduce_max(raw_output_all, axis=3)
    raw_output_all = tf.argmax(raw_output_all, axis=3)
    pred_all = tf.expand_dims(raw_output_all, axis=3)

    raw_edge = tf.reduce_mean(tf.stack([edge_out2]), axis=0)
    head_output, tail_output = tf.unstack(raw_edge, num=2, axis=0)
    tail_output_rev = tf.reverse(tail_output, axis=[1])
    raw_edge_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
    raw_edge_all = tf.expand_dims(raw_edge_all, axis=0)
    pred_edge = tf.sigmoid(raw_edge_all)
    res_edge = tf.cast(tf.greater(pred_edge, 0.5), tf.int32)

    # Which variables to load.
    restore_var = tf.compat.v1.global_variables()
    # Set up tf session and initialize variables. 
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 使用するメモリの割合を設定
    sess = tf.compat.v1.Session(config=config)
    init = tf.compat.v1.global_variables_initializer()
    
    sess.run(init)
    sess.run(tf.compat.v1.local_variables_initializer())
    
    # Load weights.
    loader = tf.compat.v1.train.Saver(var_list=restore_var)
    if RESTORE_FROM is not None:
        if load(loader, sess, RESTORE_FROM):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            return

    # evaluate prosessing
    parsing_dir = './output/cihp_parsing_maps'
    if not os.path.exists(parsing_dir):
        os.makedirs(parsing_dir)
    edge_dir = './output/cihp_edge_maps'
    if not os.path.exists(edge_dir):
        os.makedirs(edge_dir)

    for step in range(NUM_STEPS):
        print(step)
        try:
            parsing_, scores, edge_ = sess.run([pred_all, pred_scores, pred_edge])
        except tf.errors.OutOfRangeError:
            break

        if step % 1 == 0:
            print('step {:d}'.format(step))
            print(image_files[step])
        img_split = image_files[step].split('/')
        img_id = img_split[-1][:-4]

        msk = decode_labels(parsing_, num_classes=N_CLASSES)

        parsing_im = Image.fromarray(msk[0])
        parsing_im.save('{}/{}_vis.png'.format(parsing_dir, img_id))
        cv2.imwrite('{}/{}.png'.format(parsing_dir, img_id), parsing_[0,:,:,0])
        cv2.imwrite('{}/{}.png'.format(edge_dir, img_id), edge_[0,:,:,0] * 255)
        print("here")

if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()

