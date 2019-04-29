# Script for visualising the SELD output.
#
# NOTE: Make sure to use the appropriate backend for the matplotlib based on your OS

import os
import sys
sys.path.append(os.path.join(sys.path[0], '..'))

import librosa.display
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plot
import numpy as np

from metrics import cls_feature_class, evaluation_metrics
from utils.utilities import event_labels, ix_to_lb

plot.switch_backend('Qt5Agg')
# plot.switch_backend('TkAgg')


def collect_classwise_data(_in_dict):
    _out_dict = {}
    for _key in _in_dict.keys():
        for _seld in _in_dict[_key]:
            if _seld[0] not in _out_dict:
                _out_dict[_seld[0]] = []
            _out_dict[_seld[0]].append([_key, _seld[0], _seld[1], _seld[2]])
    return _out_dict


def plot_func(plot_data, hop_len_s, ind, plot_x_ax=False):
    cmap = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'salmon', 'lime', 'dodgerblue', 'brown']
    for class_ind in plot_data.keys():
        time_ax = np.array(plot_data[class_ind])[:, 0] *hop_len_s
        y_ax = np.array(plot_data[class_ind])[:, ind]
        plot.plot(time_ax, y_ax, marker='s', color=cmap[class_ind], linestyle='None', markersize=8)
    plot.grid()
    plot.xlim([0, 60])
    if not plot_x_ax:
        plot.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False, #'off',  # ticks along the bottom edge are off
            top=False, #'off',  # ticks along the top edge are off
            labelbottom=False) #'off')  # labels along the bottom edge are off

    if ind == 1:
        plot.yticks(np.arange(11), event_labels)#, rotation=-45, rotation_mode='anchor')
    elif ind == 2:
        plot.yticks(np.arange(-180, 185, 40))#, rotation=-45, rotation_mode='anchor')
    elif ind == 3:
        plot.yticks(np.arange(-50, 55, 10))#, rotation=-45, rotation_mode='anchor')

    

# --------------------------------- MAIN SCRIPT STARTS HERE -----------------------------------------

# fixed hoplength of 0.02 seconds for evaluation
hop_s = 0.02

# output format file to visualize
pred = '/vol/vssp/msos/YinC/workspace/DCASE2019/task3/mycode_v10/appendixes/submissions/two_staged_eval/model_pretrained_CRNN10_regr_mic_seed_10/test/split1_ir0_ov1_7.csv'

# path of reference audio directory for visualizing the spectrogram and description directory for
# visualizing the reference
# Note: The code finds out the audio filename from the predicted filename automatically
ref_dir = '/vol/vssp/AP_datasets/audio/dcase2019/task3/dataset_root/metadata_dev/'
aud_dir = '/vol/vssp/AP_datasets/audio/dcase2019/task3/dataset_root/foa_dev/'

# load the predicted output format
pred_dict = evaluation_metrics.load_output_format_file(pred)

# load the reference output format
feat_cls = cls_feature_class.FeatureClass()
ref_filename = os.path.basename(pred)
ref_desc_dict = feat_cls.read_desc_file(os.path.join(ref_dir, ref_filename), in_sec=True)
ref_dict = evaluation_metrics.description_file_to_output_format(ref_desc_dict, feat_cls.get_classes(), hop_s)


pred_data = collect_classwise_data(pred_dict)
ref_data = collect_classwise_data(ref_dict)

nb_classes = len(feat_cls.get_classes())

# load the audio and extract spectrogram
ref_filename = os.path.basename(pred).replace('.csv', '.wav')
audio, fs = feat_cls._load_audio(os.path.join(aud_dir, ref_filename))
stft = np.abs(np.squeeze(feat_cls._spectrogram(audio[:, :1])))
stft = librosa.amplitude_to_db(stft, ref=np.max)

# plot.figure()
# gs = gridspec.GridSpec(4, 4)
# ax0 = plot.subplot(gs[0, 1:3]), librosa.display.specshow(stft.T, sr=fs, x_axis='time', y_axis='linear'), plot.title('Spectrogram')
# ax1 = plot.subplot(gs[1, :2]), plot_func(ref_data, hop_s, ind=1), plot.ylim([-1, nb_classes + 1]), plot.title('SED reference')
# ax2 = plot.subplot(gs[1, 2:]), plot_func(pred_data, hop_s, ind=1), plot.ylim([-1, nb_classes + 1]), plot.title('SED predicted')
# ax3 = plot.subplot(gs[2, :2]), plot_func(ref_data, hop_s, ind=2), plot.ylim([-190, 190]), plot.title('Azimuth DOA reference')
# ax4 = plot.subplot(gs[2, 2:]), plot_func(pred_data, hop_s, ind=2), plot.ylim([-190, 190]), plot.title('Azimuth DOA predicted')
# ax5 = plot.subplot(gs[3, :2]), plot_func(ref_data, hop_s, ind=3, plot_x_ax=True), plot.ylim([-50, 50]), plot.title('Elevation DOA reference')
# ax6 = plot.subplot(gs[3, 2:]), plot_func(pred_data, hop_s, ind=3, plot_x_ax=True), plot.ylim([-50, 50]), plot.title('Elevation DOA predicted')
# ax_lst = [ax0, ax1, ax2, ax3, ax4, ax5, ax6]
# plot.show()

# fig = plot.figure(figsize=(16, 6))
# gs = gridspec.GridSpec(2, 2)
# ax0 = plot.subplot(gs[0, 0]), plot_func(ref_data, hop_s, ind=1), plot.ylim([-1, nb_classes]), plot.title('SED Gound Truth')
# ax1 = plot.subplot(gs[0, 1]), plot_func(pred_data, hop_s, ind=1), plot.ylim([-1, nb_classes]), plot.title('SED Predictions')
# ax2 = plot.subplot(gs[1, 0]), plot_func(ref_data, hop_s, ind=2, plot_x_ax=True), plot.ylim([-190, 190]), plot.title('Azimuth DOA Gound Truth')
# ax3 = plot.subplot(gs[1, 1]), plot_func(pred_data, hop_s, ind=2, plot_x_ax=True), plot.ylim([-190, 190]), plot.title('Azimuth DOA Predictions')
# # ax2 = plot.subplot(gs[1, 0]), plot_func(ref_data, hop_s, ind=3, plot_x_ax=True), plot.ylim([-50, 50]), plot.title('Elevation DOA Gound Truth')
# # ax3 = plot.subplot(gs[1, 1]), plot_func(pred_data, hop_s, ind=3, plot_x_ax=True), plot.ylim([-50, 50]), plot.title('Elevation DOA Predictions')
# ax_lst = [ax0, ax1, ax2, ax3]
# plot.tight_layout()
# plot.savefig('./appendixes/figures/sed_doa_results.eps')
# plot.show()

font = {'fontname':'Times New Roman'}
fig = plot.figure(figsize=(16, 9))
gs = gridspec.GridSpec(2, 1)
# ax0 = plot.subplot(gs[0, 0]), plot_func(ref_data, hop_s, ind=1), plot.ylim([-1, nb_classes]), plot.title('SED Gound Truth')
# ax1 = plot.subplot(gs[0, 1]), plot_func(pred_data, hop_s, ind=1), plot.ylim([-1, nb_classes]), plot.title('SED Predictions')
ax2 = plot.subplot(gs[0, 0]), plot_func(ref_data, hop_s, ind=2, plot_x_ax=True), plot.ylim([-190, 190]), plot.title('Gound Truth', **font)
ax3 = plot.subplot(gs[1, 0]), plot_func(pred_data, hop_s, ind=2, plot_x_ax=True), plot.ylim([-190, 190]), plot.title('Predictions', **font)
# ax2 = plot.subplot(gs[1, 0]), plot_func(ref_data, hop_s, ind=3, plot_x_ax=True), plot.ylim([-50, 50]), plot.title('Elevation DOA Gound Truth')
# ax3 = plot.subplot(gs[1, 1]), plot_func(pred_data, hop_s, ind=3, plot_x_ax=True), plot.ylim([-50, 50]), plot.title('Elevation DOA Predictions')
ax_lst = [ax2, ax3]
plot.xlabel('Time in seconds', **font)
fig.text(0.01, 0.55, 'SED in colors and Azimuth angles in degrees', va='center', rotation='vertical', **font)
plot.rcParams.update({'font.size': 30})
plot.tight_layout(pad=1.4)
plot.savefig('./appendixes/figures/sed_doa_results.eps')
plot.show()
