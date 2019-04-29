#
# Implements the core metrics from sound event detection evaluation module http://tut-arg.github.io/sed_eval/ and
# The DOA metrics are explained in the SELDnet paper
#
# This script has MIT license
#

import numpy as np
from scipy.optimize import linear_sum_assignment
from IPython import embed
eps = np.finfo(np.float).eps


##########################################################################################
# SELD scoring functions - class implementation
#
# NOTE: Supports only one-hot labels for both SED and DOA. Doesnt work for baseline method
# directly, since it estimated DOA in regression approach. Check below the class for
# one shot (function) implementations of all metrics. The function implementation has
# support for both one-hot labels and regression values of DOA estimation.
##########################################################################################

class SELDMetrics(object):
    def __init__(self, nb_frames_1s=None, data_gen=None):
        # SED params
        self._S = 0
        self._D = 0
        self._I = 0
        self._TP = 0
        self._Nref = 0
        self._Nsys = 0
        self._block_size = nb_frames_1s

        # DOA params
        self._doa_loss_pred_cnt = 0
        self._nb_frames = 0

        self._doa_loss_pred = 0
        self._nb_good_pks = 0

        self._data_gen = data_gen

        self._less_est_cnt, self._less_est_frame_cnt = 0, 0
        self._more_est_cnt, self._more_est_frame_cnt = 0, 0

    def f1_overall_framewise(self, O, T):
        TP = ((2 * T - O) == 1).sum()
        Nref, Nsys = T.sum(), O.sum()
        self._TP += TP
        self._Nref += Nref
        self._Nsys += Nsys

    def er_overall_framewise(self, O, T):
        FP = np.logical_and(T == 0, O == 1).sum(1)
        FN = np.logical_and(T == 1, O == 0).sum(1)
        S = np.minimum(FP, FN).sum()
        D = np.maximum(0, FN - FP).sum()
        I = np.maximum(0, FP - FN).sum()
        self._S += S
        self._D += D
        self._I += I

    def f1_overall_1sec(self, O, T):
        new_size = int(np.ceil(O.shape[0] / self._block_size))
        O_block = np.zeros((new_size, O.shape[1]))
        T_block = np.zeros((new_size, O.shape[1]))
        for i in range(0, new_size):
            O_block[i, :] = np.max(O[int(i * self._block_size):int(i * self._block_size + self._block_size - 1), :], axis=0)
            T_block[i, :] = np.max(T[int(i * self._block_size):int(i * self._block_size + self._block_size - 1), :], axis=0)
        return self.f1_overall_framewise(O_block, T_block)

    def er_overall_1sec(self, O, T):
        new_size = int(O.shape[0] / self._block_size)
        O_block = np.zeros((new_size, O.shape[1]))
        T_block = np.zeros((new_size, O.shape[1]))
        for i in range(0, new_size):
            O_block[i, :] = np.max(O[int(i * self._block_size):int(i * self._block_size + self._block_size - 1), :], axis=0)
            T_block[i, :] = np.max(T[int(i * self._block_size):int(i * self._block_size + self._block_size - 1), :], axis=0)
        return self.er_overall_framewise(O_block, T_block)

    def update_sed_scores(self, pred, gt):
        """
        Computes SED metrics for one second segments

        :param pred: predicted matrix of dimension [nb_frames, nb_classes], with 1 when sound event is active else 0
        :param gt:  reference matrix of dimension [nb_frames, nb_classes], with 1 when sound event is active else 0
        :param nb_frames_1s: integer, number of frames in one second
        :return:
        """
        self.f1_overall_1sec(pred, gt)
        self.er_overall_1sec(pred, gt)

    def compute_sed_scores(self):
        ER = (self._S + self._D + self._I) / (self._Nref + 0.0)

        prec = float(self._TP) / float(self._Nsys + eps)
        recall = float(self._TP) / float(self._Nref + eps)
        F = 2 * prec * recall / (prec + recall + eps)

        return ER, F

    def update_doa_scores(self, pred_doa_thresholded, gt_doa):
        '''
        Compute DOA metrics when DOA is estimated using classification approach

        :param pred_doa_thresholded: predicted results of dimension [nb_frames, nb_classes, nb_azi*nb_ele],
                                    with value 1 when sound event active, else 0
        :param gt_doa: reference results of dimension [nb_frames, nb_classes, nb_azi*nb_ele],
                        with value 1 when sound event active, else 0
        :param data_gen_test: feature or data generator class

        :return: DOA metrics

        '''
        self._doa_loss_pred_cnt += np.sum(pred_doa_thresholded)
        self._nb_frames += pred_doa_thresholded.shape[0]

        for frame in range(pred_doa_thresholded.shape[0]):
            nb_gt_peaks = int(np.sum(gt_doa[frame, :]))
            nb_pred_peaks = int(np.sum(pred_doa_thresholded[frame, :]))

            # good_frame_cnt includes frames where the nb active sources were zero in both groundtruth and prediction
            if nb_gt_peaks == nb_pred_peaks:
                self._nb_good_pks += 1
            elif nb_gt_peaks > nb_pred_peaks:
                self._less_est_frame_cnt += 1
                self._less_est_cnt += (nb_gt_peaks - nb_pred_peaks)
            elif nb_pred_peaks > nb_gt_peaks:
                self._more_est_frame_cnt += 1
                self._more_est_cnt += (nb_pred_peaks - nb_gt_peaks)

            # when nb_ref_doa > nb_estimated_doa, ignores the extra ref doas and scores only the nearest matching doas
            # similarly, when nb_estimated_doa > nb_ref_doa, ignores the extra estimated doa and scores the remaining matching doas
            if nb_gt_peaks and nb_pred_peaks:
                pred_ind = np.where(pred_doa_thresholded[frame] == 1)[1]
                pred_list_rad = np.array(self._data_gen .get_matrix_index(pred_ind)) * np.pi / 180

                gt_ind = np.where(gt_doa[frame] == 1)[1]
                gt_list_rad = np.array(self._data_gen .get_matrix_index(gt_ind)) * np.pi / 180

                frame_dist = distance_between_gt_pred(gt_list_rad.T, pred_list_rad.T)
                self._doa_loss_pred += frame_dist

    def compute_doa_scores(self):
        doa_error = self._doa_loss_pred / self._doa_loss_pred_cnt
        frame_recall = self._nb_good_pks / float(self._nb_frames)
        return doa_error, frame_recall

    def reset(self):
        # SED params
        self._S = 0
        self._D = 0
        self._I = 0
        self._TP = 0
        self._Nref = 0
        self._Nsys = 0

        # DOA params
        self._doa_loss_pred_cnt = 0
        self._nb_frames = 0

        self._doa_loss_pred = 0
        self._nb_good_pks = 0

        self._less_est_cnt, self._less_est_frame_cnt = 0, 0
        self._more_est_cnt, self._more_est_frame_cnt = 0, 0


###############################################################
# SED scoring functions
###############################################################


def reshape_3Dto2D(A):
    return A.reshape(A.shape[0] * A.shape[1], A.shape[2])


def f1_overall_framewise(O, T):
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)
    TP = ((2 * T - O) == 1).sum()
    Nref, Nsys = T.sum(), O.sum()

    prec = float(TP) / float(Nsys + eps)
    recall = float(TP) / float(Nref + eps)
    f1_score = 2 * prec * recall / (prec + recall + eps)
    return f1_score


def er_overall_framewise(O, T):
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)

    FP = np.logical_and(T == 0, O == 1).sum(1)
    FN = np.logical_and(T == 1, O == 0).sum(1)

    S = np.minimum(FP, FN).sum()
    D = np.maximum(0, FN-FP).sum()
    I = np.maximum(0, FP-FN).sum()

    Nref = T.sum()
    ER = (S+D+I) / (Nref + 0.0)
    return ER


def f1_overall_1sec(O, T, block_size):
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)
    new_size = int(np.ceil(O.shape[0] / block_size))
    O_block = np.zeros((new_size, O.shape[1]))
    T_block = np.zeros((new_size, O.shape[1]))
    for i in range(0, new_size):
        O_block[i, :] = np.max(O[int(i * block_size):int(i * block_size + block_size - 1), :], axis=0)
        T_block[i, :] = np.max(T[int(i * block_size):int(i * block_size + block_size - 1), :], axis=0)
    return f1_overall_framewise(O_block, T_block)


def er_overall_1sec(O, T, block_size):
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)
    new_size = int(O.shape[0] / (block_size))
    O_block = np.zeros((new_size, O.shape[1]))
    T_block = np.zeros((new_size, O.shape[1]))
    for i in range(0, new_size):
        O_block[i, :] = np.max(O[int(i * block_size):int(i * block_size + block_size - 1), :], axis=0)
        T_block[i, :] = np.max(T[int(i * block_size):int(i * block_size + block_size - 1), :], axis=0)
    return er_overall_framewise(O_block, T_block)


def compute_sed_scores(pred, gt, nb_frames_1s):
    """
    Computes SED metrics for one second segments

    :param pred: predicted matrix of dimension [nb_frames, nb_classes], with 1 when sound event is active else 0
    :param gt:  reference matrix of dimension [nb_frames, nb_classes], with 1 when sound event is active else 0
    :param nb_frames_1s: integer, number of frames in one second
    :return:
    """
    f1o = f1_overall_1sec(pred, gt, nb_frames_1s)
    ero = er_overall_1sec(pred, gt, nb_frames_1s)
    scores = [ero, f1o]
    return scores


###############################################################
# DOA scoring functions
###############################################################


def compute_doa_scores_regr(pred_doa_rad, gt_doa_rad, pred_sed, gt_sed):
    """
        Compute DOA metrics when DOA is estimated using regression approach

    :param pred_doa_rad: predicted doa_labels is of dimension [nb_frames, 2*nb_classes],
                        nb_classes each for azimuth and elevation angles,
                        if active, the DOA values will be in RADIANS, else, it will contain default doa values
    :param gt_doa_rad: reference doa_labels is of dimension [nb_frames, 2*nb_classes],
                    nb_classes each for azimuth and elevation angles,
                    if active, the DOA values will be in RADIANS, else, it will contain default doa values
    :param pred_sed: predicted sed label of dimension [nb_frames, nb_classes] which is 1 for active sound event else zero
    :param gt_sed: reference sed label of dimension [nb_frames, nb_classes] which is 1 for active sound event else zero
    :return:
    """

    nb_src_gt_list = np.zeros(gt_doa_rad.shape[0]).astype(int)
    nb_src_pred_list = np.zeros(gt_doa_rad.shape[0]).astype(int)
    good_frame_cnt = 0
    doa_loss_pred = 0.0
    nb_sed = gt_sed.shape[-1]

    less_est_cnt, less_est_frame_cnt = 0, 0
    more_est_cnt, more_est_frame_cnt = 0, 0

    for frame_cnt, sed_frame in enumerate(gt_sed):
        nb_src_gt_list[frame_cnt] = int(np.sum(sed_frame))
        nb_src_pred_list[frame_cnt] = int(np.sum(pred_sed[frame_cnt]))

        # good_frame_cnt includes frames where the nb active sources were zero in both groundtruth and prediction
        if nb_src_gt_list[frame_cnt] == nb_src_pred_list[frame_cnt]:
            good_frame_cnt = good_frame_cnt + 1
        elif nb_src_gt_list[frame_cnt] > nb_src_pred_list[frame_cnt]:
            less_est_cnt = less_est_cnt + nb_src_gt_list[frame_cnt] - nb_src_pred_list[frame_cnt]
            less_est_frame_cnt = less_est_frame_cnt + 1
        elif nb_src_gt_list[frame_cnt] < nb_src_pred_list[frame_cnt]:
            more_est_cnt = more_est_cnt + nb_src_pred_list[frame_cnt] - nb_src_gt_list[frame_cnt]
            more_est_frame_cnt = more_est_frame_cnt + 1

        # when nb_ref_doa > nb_estimated_doa, ignores the extra ref doas and scores only the nearest matching doas
        # similarly, when nb_estimated_doa > nb_ref_doa, ignores the extra estimated doa and scores the remaining matching doas
        if nb_src_gt_list[frame_cnt] and nb_src_pred_list[frame_cnt]:
            # DOA Loss with respect to predicted confidence
            sed_frame_gt = gt_sed[frame_cnt]
            doa_frame_gt_azi = gt_doa_rad[frame_cnt][:nb_sed][sed_frame_gt == 1]
            doa_frame_gt_ele = gt_doa_rad[frame_cnt][nb_sed:][sed_frame_gt == 1]

            sed_frame_pred = pred_sed[frame_cnt]
            doa_frame_pred_azi = pred_doa_rad[frame_cnt][:nb_sed][sed_frame_pred == 1]
            doa_frame_pred_ele = pred_doa_rad[frame_cnt][nb_sed:][sed_frame_pred == 1]
            doa_loss_pred += distance_between_gt_pred(np.vstack((doa_frame_gt_azi, doa_frame_gt_ele)).T,
                                                      np.vstack((doa_frame_pred_azi, doa_frame_pred_ele)).T)

    doa_loss_pred_cnt = np.sum(nb_src_pred_list)
    if doa_loss_pred_cnt:
        doa_loss_pred /= doa_loss_pred_cnt

    frame_recall = good_frame_cnt / float(gt_sed.shape[0])
    er_metric = [doa_loss_pred, frame_recall, doa_loss_pred_cnt, good_frame_cnt, more_est_cnt, less_est_cnt]
    return er_metric


def compute_doa_scores_clas(pred_doa_thresholded, gt_doa, get_doas):
    '''
    Compute DOA metrics when DOA is estimated using classification approach

    :param pred_doa_thresholded: predicted results of dimension [nb_frames, nb_classes, nb_azi*nb_ele],
                                with value 1 when sound event active, else 0
    :param gt_doa: reference results of dimension [nb_frames, nb_classes, nb_azi*nb_ele],
                    with value 1 when sound event active, else 0
    :param get_doas: convert doa indexes to doa degrees

    :return: DOA metrics

    '''
    doa_loss_pred_cnt = np.sum(pred_doa_thresholded)

    doa_loss_pred = 0
    nb_good_pks = 0

    less_est_cnt, less_est_frame_cnt = 0, 0
    more_est_cnt, more_est_frame_cnt = 0, 0

    for frame in range(pred_doa_thresholded.shape[0]):
        nb_gt_peaks = int(np.sum(gt_doa[frame, :]))
        nb_pred_peaks = int(np.sum(pred_doa_thresholded[frame, :]))

        # good_frame_cnt includes frames where the nb active sources were zero in both groundtruth and prediction
        if nb_gt_peaks == nb_pred_peaks:
            nb_good_pks += 1
        elif nb_gt_peaks > nb_pred_peaks:
            less_est_frame_cnt += 1
            less_est_cnt += (nb_gt_peaks - nb_pred_peaks)
        elif nb_pred_peaks > nb_gt_peaks:
            more_est_frame_cnt += 1
            more_est_cnt += (nb_pred_peaks - nb_gt_peaks)

        # when nb_ref_doa > nb_estimated_doa, ignores the extra ref doas and scores only the nearest matching doas
        # similarly, when nb_estimated_doa > nb_ref_doa, ignores the extra estimated doa and scores the remaining matching doas
        if nb_gt_peaks and nb_pred_peaks:

            pred_ind = np.where(pred_doa_thresholded[frame] == 1)[0]
            pred_list_rad = np.array(get_doas(pred_ind)) * np.pi / 180

            gt_ind = np.where(gt_doa[frame] == 1)[0]
            gt_list_rad = np.array(get_doas(gt_ind)) * np.pi / 180

            frame_dist = distance_between_gt_pred(gt_list_rad, pred_list_rad)
            doa_loss_pred += frame_dist

    if doa_loss_pred_cnt:
        doa_loss_pred /= doa_loss_pred_cnt

    frame_recall = nb_good_pks / float(pred_doa_thresholded.shape[0])
    er_metric = [doa_loss_pred, frame_recall, doa_loss_pred_cnt, nb_good_pks, more_est_cnt, less_est_cnt]
    return er_metric


def distance_between_gt_pred(gt_list_rad, pred_list_rad):
    """
    Shortest distance between two sets of spherical coordinates. Given a set of groundtruth spherical coordinates,
     and its respective predicted coordinates, we calculate the spherical distance between each of the spherical
     coordinate pairs resulting in a matrix of distances, where one axis represents the number of groundtruth
     coordinates and the other the predicted coordinates. The number of estimated peaks need not be the same as in
     groundtruth, thus the distance matrix is not always a square matrix. We use the hungarian algorithm to find the
     least cost in this distance matrix.

    :param gt_list_rad: list of ground-truth spherical coordinates
    :param pred_list_rad: list of predicted spherical coordinates
    :return: cost -  distance
    :return: less - number of DOA's missed
    :return: extra - number of DOA's over-estimated
    """
    gt_len, pred_len = gt_list_rad.shape[0], pred_list_rad.shape[0]
    ind_pairs = np.array([[x, y] for y in range(pred_len) for x in range(gt_len)])
    cost_mat = np.zeros((gt_len, pred_len))

    # Slow implementation
    # cost_mat = np.zeros((gt_len, pred_len))
    # for gt_cnt, gt in enumerate(gt_list_rad):
    #     for pred_cnt, pred in enumerate(pred_list_rad):
    #         cost_mat[gt_cnt, pred_cnt] = distance_between_spherical_coordinates_rad(gt, pred)

    # Fast implementation
    if gt_len and pred_len:
        az1, ele1, az2, ele2 = gt_list_rad[ind_pairs[:, 0], 0], gt_list_rad[ind_pairs[:, 0], 1], \
                               pred_list_rad[ind_pairs[:, 1], 0], pred_list_rad[ind_pairs[:, 1], 1]
        cost_mat[ind_pairs[:, 0], ind_pairs[:, 1]] = distance_between_spherical_coordinates_rad(az1, ele1, az2, ele2)

    row_ind, col_ind = linear_sum_assignment(cost_mat)
    cost = cost_mat[row_ind, col_ind].sum()
    return cost


def distance_between_spherical_coordinates_rad(az1, ele1, az2, ele2):
    """
    Angular distance between two spherical coordinates
    MORE: https://en.wikipedia.org/wiki/Great-circle_distance

    :return: angular distance in degrees
    """
    dist = np.sin(ele1) * np.sin(ele2) + np.cos(ele1) * np.cos(ele2) * np.cos(np.abs(az1 - az2))
    # Making sure the dist values are in -1 to 1 range, else np.arccos kills the job
    dist = np.clip(dist, -1, 1)
    dist = np.arccos(dist) * 180 / np.pi
    return dist


def distance_between_cartesian_coordinates(x1, y1, z1, x2, y2, z2):
    """
    Angular distance between two cartesian coordinates
    MORE: https://en.wikipedia.org/wiki/Great-circle_distance
    Check 'From chord length' section

    :return: angular distance in degrees
    """
    dist = np.sqrt((x1-x2) ** 2 + (y1-y2) ** 2 + (z1-z2) ** 2)
    dist = 2 * np.arcsin(dist / 2.0) * 180/np.pi
    return dist


def sph2cart(azimuth, elevation, r):
    '''
    Convert spherical to cartesian coordinates

    :param azimuth: in radians
    :param elevation: in radians
    :param r: in meters
    :return: cartesian coordinates
    '''

    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return x, y, z


def cart2sph(x, y, z):
    '''
    Convert cartesian to spherical coordinates

    :param x:
    :param y:
    :param z:
    :return: azi, ele in radians and r in meters
    '''

    azimuth = np.arctan2(y,x)
    elevation = np.arctan2(z,np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return azimuth, elevation, r


###############################################################
# SELD scoring functions
###############################################################


def compute_seld_metric(sed_error, doa_error):
    """
    Compute SELD metric from sed and doa errors.

    :param sed_error: [error rate (0 to 1 range), f score (0 to 1 range)]
    :param doa_error: [doa error (in degrees), frame recall (0 to 1 range)]
    :return: seld metric result
    """
    seld_metric = np.mean([
        sed_error[0],
        1 - sed_error[1],
        doa_error[0]/180,
        1 - doa_error[1]]
        )
    return seld_metric


def compute_seld_metrics_from_output_format_dict(_pred_dict, _gt_dict, _feat_cls):
    """
        Compute SELD metrics between _gt_dict and_pred_dict in DCASE output format

    :param _pred_dict: dcase output format dict
    :param _gt_dict: dcase output format dict
    :param _feat_cls: feature or data generator class
    :return: the seld metrics
    """
    _gt_labels = output_format_dict_to_classification_labels(_gt_dict, _feat_cls)
    _pred_labels = output_format_dict_to_classification_labels(_pred_dict, _feat_cls)

    _er, _f = compute_sed_scores(_pred_labels.max(2), _gt_labels.max(2), _feat_cls.nb_frames_1s())
    _doa_err, _frame_recall, d1, d2, d3, d4 = compute_doa_scores_clas(_pred_labels, _gt_labels, _feat_cls)
    _seld_scr = compute_seld_metric([_er, _f], [_doa_err, _frame_recall])
    return _seld_scr, _er, _f, _doa_err, _frame_recall


###############################################################
# Functions for format conversions
###############################################################

def output_format_dict_to_classification_labels(_output_dict, _feat_cls):

    _unique_classes = _feat_cls.get_classes()
    _nb_classes = len(_unique_classes)
    _azi_list, _ele_list = _feat_cls.get_azi_ele_list()
    _max_frames = _feat_cls.get_nb_frames()
    _labels = np.zeros((_max_frames, _nb_classes, len(_azi_list) * len(_ele_list)))

    for _frame_cnt in _output_dict.keys():
        if _frame_cnt < _max_frames:
            for _tmp_doa in _output_dict[_frame_cnt]:
                # Making sure the doa's are within the limits
                _tmp_doa[1] = np.clip(_tmp_doa[1], _azi_list[0], _azi_list[-1])
                _tmp_doa[2] = np.clip(_tmp_doa[2], _ele_list[0], _ele_list[-1])

                # create label
                _labels[_frame_cnt, _tmp_doa[0], int(_feat_cls.get_list_index(_tmp_doa[1], _tmp_doa[2]))] = 1

    return _labels


def regression_label_format_to_output_format(_feat_cls, _sed_labels, _doa_labels_deg):
    """
    Converts the sed (classification) and doa labels predicted in regression format to dcase output format.

    :param _feat_cls: feature or data generator class instance
    :param _sed_labels: SED labels matrix [nb_frames, nb_classes]
    :param _doa_labels_deg: DOA labels matrix [nb_frames, 2*nb_classes] in degrees
    :return: _output_dict: returns a dict containing dcase output format
    """

    _unique_classes = _feat_cls.get_classes()
    _nb_classes = len(_unique_classes)
    _azi_labels = _doa_labels_deg[:, :_nb_classes]
    _ele_labels = _doa_labels_deg[:, _nb_classes:]

    _output_dict = {}
    for _frame_ind in range(_sed_labels.shape[0]):
        _tmp_ind = np.where(_sed_labels[_frame_ind, :])
        if len(_tmp_ind[0]):
            _output_dict[_frame_ind] = []
            for _tmp_class in _tmp_ind[0]:
                _output_dict[_frame_ind].append([_tmp_class, _azi_labels[_frame_ind, _tmp_class], _ele_labels[_frame_ind, _tmp_class]])
    return _output_dict


def classification_label_format_to_output_format(_feat_cls, _labels):
    """
    Converts the seld labels predicted in classification format to dcase output format.

    :param _feat_cls: feature or data generator class instance
    :param _labels: SED labels matrix [nb_frames, nb_classes, nb_azi*nb_ele]
    :return: _output_dict: returns a dict containing dcase output format
    """
    _output_dict = {}
    for _frame_ind in range(_labels.shape[0]):
        _tmp_class_ind = np.where(_labels[_frame_ind].sum(1))
        if len(_tmp_class_ind[0]):
            _output_dict[_frame_ind] = []
            for _tmp_class in _tmp_class_ind[0]:
                _tmp_spatial_ind = np.where(_labels[_frame_ind, _tmp_class])
                for _tmp_spatial in _tmp_spatial_ind[0]:
                    _azi, _ele = _feat_cls.get_matrix_index(_tmp_spatial)
                    _output_dict[_frame_ind].append(
                        [_tmp_class, _azi, _ele])

    return _output_dict


def description_file_to_output_format(_desc_file_dict, _unique_classes, _hop_length_sec):
    """
    Reads description file in csv format. Outputs, the dcase format results in dictionary, and additionally writes it
    to the _output_file

    :param _unique_classes: unique classes dictionary, maps class name to class index
    :param _desc_file_dict: full path of the description file
    :param _hop_length_sec: hop length in seconds

    :return: _output_dict: dcase output in dicitionary format
    """

    _output_dict = {}
    for _ind, _tmp_start_sec in enumerate(_desc_file_dict['start']):
        _tmp_class = _unique_classes[_desc_file_dict['class'][_ind]]
        _tmp_azi = _desc_file_dict['azi'][_ind]
        _tmp_ele = _desc_file_dict['ele'][_ind]
        _tmp_end_sec = _desc_file_dict['end'][_ind]

        _start_frame = int(_tmp_start_sec / _hop_length_sec)
        _end_frame = int(_tmp_end_sec / _hop_length_sec)
        for _frame_ind in range(_start_frame, _end_frame + 1):
            if _frame_ind not in _output_dict:
                _output_dict[_frame_ind] = []
            _output_dict[_frame_ind].append([_tmp_class, _tmp_azi, _tmp_ele])

    return _output_dict


def load_output_format_file(_output_format_file):
    """
    Loads DCASE output format csv file and returns it in dictionary format

    :param _output_format_file: DCASE output format CSV
    :return: _output_dict: dictionary
    """
    _output_dict = {}
    _fid = open(_output_format_file, 'r')
    # next(_fid)
    for _line in _fid:
        _words = _line.strip().split(',')
        _frame_ind = int(_words[0])
        if _frame_ind not in _output_dict:
            _output_dict[_frame_ind] = []
        _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), int(_words[3])])
    _fid.close()
    return _output_dict


def write_output_format_file(_output_format_file, _output_format_dict):
    """
    Writes DCASE output format csv file, given output format dictionary

    :param _output_format_file:
    :param _output_format_dict:
    :return:
    """
    _fid = open(_output_format_file, 'w')
    # _fid.write('{},{},{},{}\n'.format('frame number with 20ms hop (int)', 'class index (int)', 'azimuth angle (int)', 'elevation angle (int)'))
    for _frame_ind in _output_format_dict.keys():
        for _value in _output_format_dict[_frame_ind]:
            _fid.write('{},{},{},{}\n'.format(int(_frame_ind), int(_value[0]), int(_value[1]), int(_value[2])))
    _fid.close()
