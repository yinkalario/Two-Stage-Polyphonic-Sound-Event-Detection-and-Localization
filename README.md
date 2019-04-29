# Polyphonic-Sound-Event-Detection-and-Localization-using-a-Two-Stage-Strategy

Sound event detection (SED) and localization refer to recognizing sound events and estimating their spatial and temporal locations. In this repo, a Two-Stage Polyphonic Sound Event Detection and Localization method is implemented using pytorch. This method is tested on the DCASE 2019 Task 3 Sound Event Localization and Detection. More description of this task can be found in http://dcase.community/challenge2019/task-sound-event-localization-and-detection.

## Citation

The code in this repo is easy to understand and implement. Please also check our baseline method in https://github.com/qiuqiangkong/dcase2019_task3.

If you found our codes are useful, please cite the following papers:

>[1] Yin Cao, Qiuqiang Kong, Turab Iqbal, Fengyan An, Wenwu Wang, Mark D. Plumbley. Polyphonic Sound Event Detection and Localization Using Two-Stage Strategy.
>Paper URL: 

>[2] Qiuqiang Kong, Yin Cao, Turab Iqbal, Yong Xu, Wenwu Wang, Mark D. Plumbley. Cross-task learning for audio tagging, sound event detection and spatial localization: DCASE 2019 baseline systems. arXiv preprint arXiv:1904.03476 (2019).
>Paper URL: https://arxiv.org/abs/1904.03476


## Dataset

The dataset can be downloaded from http://dcase.community/challenge2019/task-sound-event-localization-and-detection. This dataset contains 400 audio recordings splitted into 4 folds. Two formats of audios are givin: 1) First-Order of Ambisonics; 2) tetrahedral microphone array. There are 11 kinds of isolated sound events in total. The audio recordings are mixtures of isolated sound events and natural ambient noise. The sound events, which have a polyphony of up to two, are convolved with impulse responses collected from five indoor locations.

## The method

The input features used is log mel and GCC-PHAT spectrograms, the detailed description can be found in the paper.

### Input Features

### Network Architecture
<img src="appendixes/figures/two_stage_SEDL.pdf" width="500">



