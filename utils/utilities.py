import datetime
import itertools
import logging
import os
import sys

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from librosa.display import specshow

from tqdm import tqdm

event_labels = ['knock', 'drawer', 'clearthroat', 'phone', 'keysDrop',\
    'speech', 'keyboard', 'pageturn', 'cough', 'doorslam', 'laughter']
lb_to_ix = {lb: i for i, lb in enumerate(event_labels)}
ix_to_lb = {i: lb for i, lb in enumerate(event_labels)}

azimuths = range(-180, 171, 10)
elevations = range(-40, 41, 10)
doa = [azimuths, elevations]
doa_labels = list(itertools.product(*doa))
doa_to_ix = {doa: i for i, doa in enumerate(doa_labels)}
ix_to_doa = {i: doa for i, doa in enumerate(doa_labels)}

# training_splits_dict = {1: (3,4), 2: (4,1), 3: (1,2), 4: (2,3)}
# validation_split_dict = {1: 2, 2: 3, 3: 4, 4: 1}
# testing_split_dict = {1: 1, 2: 2, 3: 3, 4: 4}

train_splits_dict = {1: [2,3,4], 2: [1,3,4], 3: [1,2,4], 4: [1,2,3], -1: [1,2,3,4]}
validation_split_dict = {1: [1], 2: [2], 3: [3], 4: [4], -1: []}
test_split_dict = {1: [1], 2: [2], 3: [3], 4: [4], -1: []}


def get_doas(indexes):
    '''
    Get multiple doas from indexes
    '''
    doas = []
    for idx in indexes:
        doas.append(ix_to_doa[idx])
    return doas


def calculate_scalar(features):

    mean = []
    std = []

    channels = features.shape[0]
    for channel in range(channels):
        feat = features[channel, :, :]
        mean.append(np.mean(feat, axis=0))
        std.append(np.std(feat, axis=0))

    mean = np.array(mean)
    std = np.array(std)
    mean = np.expand_dims(mean, axis=0)
    std = np.expand_dims(std, axis=0)
    mean = np.expand_dims(mean, axis=2)
    std = np.expand_dims(std, axis=2)

    return mean, std


def one_hot_encode(target, length):
    """Convert batches of class indices to classes of one-hot vectors."""
    target = np.array(target)
    if len(target.shape) == 0:
        one_hot_vec = np.zeros((1, length))
        one_hot_vec[0, target] = 1.0
    else:
        batch_s = target.shape[0]
        one_hot_vec = np.zeros((batch_s, length))
        for i in range(batch_s):
            one_hot_vec[i, target[i].astype(int)] = 1.0

    return one_hot_vec


def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except:
            self.handleError(record)  


def create_logging(log_dir, filemode):

    os.makedirs(log_dir, exist_ok=True)
    
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '{:04d}.log'.format(i1))):
        i1 += 1
        
    log_path = os.path.join(log_dir, '{:04d}.log'.format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        # format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # logging.getLogger('').addHandler(console)
    logging.getLogger('').addHandler(TqdmLoggingHandler())

    logging.info(datetime.datetime.now())
    logging.info('\n')

    return logging


def to_torch(x, cuda):

    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        raise Exception("Error!")

    if cuda:
        x = x.cuda()

    return x


def to_np(x):
    """
    Convert values of the model parameters to numpy.array.
    """
    return x.cpu().data.numpy()


def logging_and_writer(data_type, metrics, logging, writer=[], batch_idx=0):
    '''
    Logging to tqdm, and write to tensorboard

    Input:
        data_type: 'train' | 'valid' | 'test'
        metrics: output from evaluate function, including loss and other metrics
        logging: logging
        writer: tensorboard writer
        batch_idx: batch iteration index, only for 'train' and 'valid'
    '''

    if data_type == 'train':

        [tr_loss, tr_sed_mAP, tr_sed_scores, tr_doa_er_metric, 
            tr_seld_metric] = metrics

        logging.info('Train SELD loss: {:.3f},  Train SED loss: {:.3f},  Train DOA loss: {:.3f},  '
            'Train SED mAP(micro): {:.3f},  Train SED mAP(macro): {:.3f}'.format(
                tr_loss[0], tr_loss[1], tr_loss[2], tr_sed_mAP[0], tr_sed_mAP[1]))
        writer.add_scalar('train/SELD_loss', tr_loss[0], batch_idx)
        writer.add_scalar('train/SED_loss', tr_loss[1], batch_idx)
        writer.add_scalar('train/DOA_loss', tr_loss[2], batch_idx)
        writer.add_scalar('train/SED_mAP_micro', tr_sed_mAP[0], batch_idx)
        writer.add_scalar('train/SED_mAP_macro', tr_sed_mAP[1], batch_idx) 
       

        logging.info('Train ER: {:.3f},  Train F-score: {:.3f},  Train DOA error: {:.3f},  Train DOA frame recall: {:.3f},  Train SELD error: {:.3f}'.format(
            tr_sed_scores[0], tr_sed_scores[1], tr_doa_er_metric[0], tr_doa_er_metric[1], tr_seld_metric))
        writer.add_scalar('train/ER', tr_sed_scores[0], batch_idx)
        writer.add_scalar('train/F_score', tr_sed_scores[1], batch_idx)
        writer.add_scalar('train/DOA_error', tr_doa_er_metric[0], batch_idx)
        writer.add_scalar('train/DOA_frame_recall', tr_doa_er_metric[1], batch_idx)
        writer.add_scalar('train/SELD_error', tr_seld_metric, batch_idx)  

    elif data_type == 'valid':
        
        [train_metrics, valid_metrics] = metrics

        [tr_loss, tr_sed_mAP, tr_sed_scores, tr_doa_er_metric, 
            tr_seld_metric] = train_metrics

        [va_loss, va_sed_mAP, va_sed_scores, va_doa_er_metric, 
            va_seld_metric] = valid_metrics

        logging.info('Train SELD loss: {:.3f},  Train SED loss: {:.3f},  Train DOA loss: {:.3f},  '
            'Train SED mAP(micro): {:.3f},  Train SED mAP(macro): {:.3f}'.format(
                tr_loss[0], tr_loss[1], tr_loss[2], tr_sed_mAP[0], tr_sed_mAP[1]))
        writer.add_scalar('train/SELD_loss', tr_loss[0], batch_idx)
        writer.add_scalar('train/SED_loss', tr_loss[1], batch_idx)
        writer.add_scalar('train/DOA_loss', tr_loss[2], batch_idx)
        writer.add_scalar('train/SED_mAP_micro', tr_sed_mAP[0], batch_idx)
        writer.add_scalar('train/SED_mAP_macro', tr_sed_mAP[1], batch_idx) 

        logging.info('Valid SELD loss: {:.3f},  Valid SED loss: {:.3f},  Valid DOA loss: {:.3f},  '
            'Valid SED mAP(micro): {:.3f},  Valid SED mAP(macro): {:.3f}'.format(
                va_loss[0], va_loss[1], va_loss[2], va_sed_mAP[0], va_sed_mAP[1]))
        writer.add_scalar('valid/SELD_loss', va_loss[0], batch_idx)
        writer.add_scalar('valid/SED_loss', va_loss[1], batch_idx)
        writer.add_scalar('valid/DOA_loss', va_loss[2], batch_idx)
        writer.add_scalar('valid/SED_mAP_micro', va_sed_mAP[0], batch_idx)
        writer.add_scalar('valid/SED_mAP_macro', va_sed_mAP[1], batch_idx) 

        logging.info('Train ER: {:.3f},  Train F-score: {:.3f},  Train DOA error: {:.3f},  Train DOA frame recall: {:.3f},  Train SELD error: {:.3f}'.format(
            tr_sed_scores[0], tr_sed_scores[1], tr_doa_er_metric[0], tr_doa_er_metric[1], tr_seld_metric))
        writer.add_scalar('train/ER', tr_sed_scores[0], batch_idx)
        writer.add_scalar('train/F_score', tr_sed_scores[1], batch_idx)
        writer.add_scalar('train/DOA_error', tr_doa_er_metric[0], batch_idx)
        writer.add_scalar('train/DOA_frame_recall', tr_doa_er_metric[1], batch_idx)
        writer.add_scalar('train/SELD_error', tr_seld_metric, batch_idx)  

        logging.info('Valid ER: {:.3f},  Valid F-score: {:.3f},  Valid DOA error: {:.3f},  Valid DOA frame recall: {:.3f},  Valid SELD error: {:.3f}'.format(
            va_sed_scores[0], va_sed_scores[1], va_doa_er_metric[0], va_doa_er_metric[1], va_seld_metric))
        writer.add_scalar('valid/ER', va_sed_scores[0], batch_idx)
        writer.add_scalar('valid/F_score', va_sed_scores[1], batch_idx)
        writer.add_scalar('valid/DOA_error', va_doa_er_metric[0], batch_idx)
        writer.add_scalar('valid/DOA_frame_recall', va_doa_er_metric[1], batch_idx)
        writer.add_scalar('valid/SELD_error', va_seld_metric, batch_idx)
    
    elif data_type == 'test':

        [te_loss, te_sed_mAP, te_sed_scores, te_doa_er_metric, 
            te_seld_metric] = metrics

        logging.info('Test SELD loss: {:.3f},  Test SED loss: {:.3f},  Test DOA loss: {:.3f},  '
            'Test SED mAP(micro): {:.3f},  Test SED mAP(macro): {:.3f}'.format(
                te_loss[0], te_loss[1], te_loss[2], te_sed_mAP[0], te_sed_mAP[1]))

        logging.info('Test ER: {:.3f},  Test F-score: {:.3f},  Test DOA error: {:.3f},  Test DOA frame recall: {:.3f},  Test SELD error: {:.3f}'.format(
            te_sed_scores[0], te_sed_scores[1], te_doa_er_metric[0], te_doa_er_metric[1], te_seld_metric))


def print_evaluation(metrics):

    [te_loss, te_sed_mAP, te_sed_scores, te_doa_er_metric, 
        te_seld_metric] = metrics

    print('Test SELD loss: {:.3f},  Test SED loss: {:.3f},  Test DOA loss: {:.3f},  '
        'Test SED mAP(micro): {:.3f},  Test SED mAP(macro): {:.3f}'.format(
            te_loss[0], te_loss[1], te_loss[2], te_sed_mAP[0], te_sed_mAP[1]))

    print('Test ER: {:.3f},  Test F-score: {:.3f},  Test DOA error: {:.3f},  Test DOA frame recall: {:.3f},  Test SELD error: {:.3f}'.format(
        te_sed_scores[0], te_sed_scores[1], te_doa_er_metric[0], te_doa_er_metric[1], te_seld_metric))

