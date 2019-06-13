import os
import pdb

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, confusion_matrix

from loss import hybrid_regr_loss
from metrics import cls_feature_class, evaluation_metrics
from utils.utilities import get_doas, to_np, to_torch


def evaluate(args, data_generator, data_type, max_audio_num, task_type, model, cuda, loss_type, 
        threshold, submissions_dir=None, frames_per_1s=100, sub_frames_per_1s=50):
    '''
    Evaluate metrics for cross validation or test data

    Input:
        data_generator: data loader
        data_type: 'train' | 'valid' | 'test'
        max_audio_num: maximum audio number to evaluate the performance, None for using all clips
        task_type: 'sed_only' | 'doa_only' | 'two_staged_eval' | 'seld'
        model: nn model
        cuda: True or False to use cuda or not
        threshold: {'sed': event detection threshold,
                    'doa': doa threshold}
    Returns:
        loss_dict: {'loss': event_loss + beta*(elevation_loss + azimuth_loss),
                    'event_loss': event_loss,
                    'doa_loss': elevation_loss + azimuth_loss}
    '''

    if data_type == 'train':
        generate_func = data_generator.generate_test(data_type='train', 
            max_audio_num=max_audio_num)
    elif data_type == 'valid':
        generate_func = data_generator.generate_test(data_type='valid', 
            max_audio_num=max_audio_num)
    elif data_type == 'test':
        generate_func = data_generator.generate_test(data_type='test',
            max_audio_num=max_audio_num)

    sed_gt = []
    doa_gt = []
    sed_pred = []
    doa_pred = []

    forLoss_sed_pred = []
    forLoss_doa_pred = []

    for batch_x, batch_y_dict, batch_fn in generate_func:
        '''
        batch_size = 1
        batch_x: features
        batch_y_dict = {
            'events',       (time_len, class_num)
            'doas',         (time_len, 2*class_num) for 'regr' | 
                            (time_len, class_num, ele_num*azi_num=324) for 'clas'
            # 'distances'     (time_len, class_num)
        }
        batch_fn: filenames
        '''
        batch_x = to_torch(batch_x, cuda)
        with torch.no_grad():
            model.eval()
            output = model(batch_x)
        output['events'] = to_np(output['events'])
        output['doas'] = to_np(output['doas'])
        '''
        output = {
            'events',   (batch_size=1, time_len, class_num) 
            'doas'      (batch_size=1, time_len, 2*class_num) for 'regr' | 
                        (batch_size=1, time_len, ele_num*azi_num=324) for 'clas'
        }
        '''

        #############################################################################################################
        # save predicted sed results in 'sed_only' task
        # set output['events'] to ground truth sed in 'doa_only' task
        # load predicted sed results in 'two_staged_eval' task
        temp_hdf5_path = os.path.join(submissions_dir, batch_fn + '.h5')
        if task_type == 'sed_only':
            with h5py.File(temp_hdf5_path, 'w') as hf:
                hf.create_dataset('sed_pred', data=output['events'], dtype=np.float32)
        elif task_type == 'doa_only':
            ###set predictions is equal to ground truth
            temp = np.expand_dims(batch_y_dict['events'], axis=0)
            if output['events'].shape[1] <= temp.shape[1]:
                output['events'] = temp[:, 0: output['events'].shape[1]]
            else:
                output['events'] = np.concatenate((temp, 
                        np.zeros((1, output['events'].shape[1]-temp.shape[1], temp.shape[2]))), axis=1)
        elif task_type == 'two_staged_eval':
            with h5py.File(temp_hdf5_path, 'r') as hf:
                output['events'] = hf['sed_pred'][:]
        #############################################################################################################

        min_idx = min(batch_y_dict['events'].shape[0], output['events'].shape[1])

        sed_gt.append(batch_y_dict['events'][:min_idx])
        doa_gt.append(batch_y_dict['doas'][:min_idx])

        sed_pred.append((output['events'] > threshold['sed']).squeeze().astype(np.float32)[:min_idx])
        doa_pred.append(output['doas'].squeeze()[:min_idx])

        forLoss_sed_pred.append(output['events'].squeeze()[:min_idx])
        forLoss_doa_pred.append(output['doas'].squeeze()[:min_idx])

        ##################### for submission method evaluation ######################
        output_dict = {
            'filename': batch_fn,
            'events': (output['events']>threshold['sed']).squeeze().astype(np.float32),
            'doas': output['doas'].squeeze()}
        submit_dict = calculate_submission(output_dict, frames_per_1s, sub_frames_per_1s)
        write_submission(submit_dict, submissions_dir)
        ##############################################################################

    sed_gt = np.concatenate(sed_gt, axis=0)
    doa_gt = np.concatenate(doa_gt, axis=0)
    sed_pred = np.concatenate(sed_pred, axis=0)
    doa_pred = np.concatenate(doa_pred, axis=0)

    ###################### SED and DOA metrics, for submission method evaluation ######################
    gt_meta_dir = os.path.join(args.dataset_dir, 'metadata_dev')
    sed_scores, doa_er_metric, seld_metric = calculate_SELD_metrics(gt_meta_dir, submissions_dir, score_type='all')
    ###################################################################################################

    ## mAP
    sed_mAP_micro = average_precision_score(sed_gt, sed_pred, average='micro')
    sed_mAP_macro = average_precision_score(sed_gt, sed_pred, average='macro')
    sed_mAP = [sed_mAP_micro, sed_mAP_macro]

    ## loss
    forLoss_gt_dict = {
        'events': to_torch(sed_gt[None,:,:], cuda=False),
        'doas':   to_torch(doa_gt[None,:,:], cuda=False)
    }
    forLoss_pred_dict = {
        'events': to_torch(np.concatenate(forLoss_sed_pred, axis=0)[None,:,:], cuda=False),
        'doas':   to_torch(np.concatenate(forLoss_doa_pred, axis=0)[None,:,:], cuda=False)
    }

    seld_loss, sed_loss, doa_loss = hybrid_regr_loss(forLoss_pred_dict, forLoss_gt_dict, task_type, loss_type=loss_type)
    loss = [to_np(seld_loss), to_np(sed_loss), to_np(doa_loss)]

    metrics = [loss, sed_mAP, sed_scores, doa_er_metric, seld_metric]

    # torch.cuda.empty_cache()

    return  metrics 


def calculate_submission(output_dict, frames_per_1s, sub_frames_per_1s=50):
    '''
    Interoplate tensor to length of 20ms
    '''
    
    output_dict['events'] = interp_tensor(output_dict['events'], frames_per_1s, sub_frames_per_1s)
    output_dict['doas'] = interp_tensor(output_dict['doas'], frames_per_1s, sub_frames_per_1s)

    return output_dict


def interp_tensor(tensor, frames_per_1s, sub_frames_per_1s=50):
    '''
    Interpolate tensor
    
    Args:
        tensor: (time_steps, event_class_num)
        frames_per_1s: submission frames_per_1s
        sub_frames_per_1s: submission frames per 1 s
    '''
    ratio = 1.0 * sub_frames_per_1s / frames_per_1s

    new_len = int(np.around(ratio * tensor.shape[0]))
    new_tensor = np.zeros((new_len, tensor.shape[1]))

    for n in range(new_len):
        new_tensor[n] = tensor[int(np.around(n / ratio))]
    
    return new_tensor


def write_submission(dict, submissions_dir):
    '''
    Write predicted result to submission csv files

    Args:
        dict={
            'filename': file name,
            'events': (time_len, class_num)
            'doas': (time_len, 2*class_num) for 'regr' | 
                    (time_len, ele_num*azi_num=324) for 'clas'
        }
    '''

    fn = '{}.csv'.format(dict['filename'])
    file_path = os.path.join(submissions_dir, fn)

    with open(file_path, 'w') as f:
        for n in range(dict['events'].shape[0]):
            event_indexes = np.where(dict['events'][n]==1.0)[0]
            azi = np.around(dict['doas'][n, event_indexes] * 180 / np.pi, 
                decimals=-1)
            ele = np.around(dict['doas'][n, event_indexes+dict['events'].shape[1]] * 180 / np.pi, 
                decimals=-1)
            for idx, k in enumerate(event_indexes):
                f.write('{},{},{},{}\n'.format(n, k, int(azi[idx]), int(ele[idx])))


def get_nb_files(_pred_file_list, _group='split'):
    '''Get attributes number
    https://github.com/sharathadavanne/seld-dcase2019/blob/master/calculate_SELD_metrics.py
    '''
    _group_ind = {'split': 5, 'ir': 9, 'ov': 13}
    _cnt_dict = {}
    for _filename in _pred_file_list:

        if _group == 'all':
            _ind = 0
        else:
            _ind = int(_filename[_group_ind[_group]])

        if _ind not in _cnt_dict:
            _cnt_dict[_ind] = []
        _cnt_dict[_ind].append(_filename)

    return _cnt_dict


def calculate_SELD_metrics(gt_meta_dir, pred_meta_dir, score_type):
    '''Calculate metrics using official tool. This part of code is modified from:
    https://github.com/sharathadavanne/seld-dcase2019/blob/master/calculate_SELD_metrics.py
    
    Args:
      gt_meta_dir: ground truth meta directory. 
      pred_meta_dir: prediction meta directory.
      score_type: 'all', 'split', 'ov', 'ir'
      
    Returns:
      metrics: dict
    '''
    
    # Load feature class
    feat_cls = cls_feature_class.FeatureClass()

    # collect gt files info
    # gt_meta_files = [fn for fn in os.listdir(gt_meta_dir) if fn.endswith('.csv') and not fn.startswith('.')]

    # collect pred files info
    pred_meta_files = [fn for fn in os.listdir(pred_meta_dir) if fn.endswith('.csv') and not fn.startswith('.')]

    # Load evaluation metric class
    eval = evaluation_metrics.SELDMetrics(
        nb_frames_1s=feat_cls.nb_frames_1s(), data_gen=feat_cls)
    
    # Calculate scores for different splits, overlapping sound events, and impulse responses (reverberant scenes)
    # score_type = 'all', 'split', 'ov', 'ir'
    split_cnt_dict = get_nb_files(pred_meta_files, _group=score_type)

    sed_error_rate = []
    sed_f1_score = []
    doa_error = []
    doa_frame_recall = []
    seld_metric = []

    # Calculate scores across files for a given score_type
    for split_key in np.sort(list(split_cnt_dict)):
        eval.reset()    # Reset the evaluation metric parameters
        for _, pred_file in enumerate(split_cnt_dict[split_key]):
            # Load predicted output format file
            pred_dict = evaluation_metrics.load_output_format_file(os.path.join(pred_meta_dir, pred_file))

            # Load reference description file
            gt_desc_file_dict = feat_cls.read_desc_file(os.path.join(gt_meta_dir, pred_file.replace('.npy', '.csv')))

            # Generate classification labels for SELD
            gt_labels = feat_cls.get_clas_labels_for_file(gt_desc_file_dict)
            pred_labels = evaluation_metrics.output_format_dict_to_classification_labels(pred_dict, feat_cls)

            # Calculated SED and DOA scores
            eval.update_sed_scores(pred_labels.max(2), gt_labels.max(2))
            eval.update_doa_scores(pred_labels, gt_labels)

        # Overall SED and DOA scores
        sed_er, sed_f1 = eval.compute_sed_scores()
        doa_err, doa_fr = eval.compute_doa_scores()
        seld_metr = evaluation_metrics.compute_seld_metric(
            [sed_er, sed_f1], [doa_err, doa_fr])

        sed_error_rate.append(sed_er)
        sed_f1_score.append(sed_f1)
        doa_error.append(doa_err)
        doa_frame_recall.append(doa_fr)
        seld_metric.append(seld_metr)

    sed_scores = [sed_error_rate, sed_f1_score]
    doa_er_metric = [doa_error, doa_frame_recall]

    sed_scores = np.array(sed_scores).squeeze()
    doa_er_metric = np.array(doa_er_metric).squeeze()
    seld_metric = np.array(seld_metric).squeeze()

    return sed_scores, doa_er_metric, seld_metric
