#!/bin/bash

# Data directory
DATASET_DIR='/vol/vssp/AP_datasets/audio/dcase2019/task3/dataset_root/'

# Feature directory
FEATURE_DIR='/vol/vssp/msos/YinC/workspace/Dataset_Features/DCASE2019/task3/'

# Workspace
WORKSPACE='/vol/vssp/msos/YinC/workspace/DCASE2019/task3/mycode_v11/'
cd $WORKSPACE


########### Hyper-parameters ###########
FEATURE_TYPE='logmelgcc'
AUDIO_TYPE='foa'
FOLD=1

# TASK_TYPE: 'sed_only' | 'doa_only' | 'two_staged_eval' | 'seld'
TASK_TYPE='sed_only'

# GPU number
GPU_ID=2

# Extract Features
python utils/feature_extractor.py --dataset_dir=$DATASET_DIR --feature_dir=$FEATURE_DIR --feature_type=$FEATURE_TYPE --data_type='dev' --audio_type=$AUDIO_TYPE

############ Development ############
# train
# -W ignore 
SEED=10
CUDA_VISIBLE_DEVICES=$GPU_ID python ${WORKSPACE}main.py train --workspace=$WORKSPACE --feature_dir=$FEATURE_DIR --feature_type=$FEATURE_TYPE --audio_type=$AUDIO_TYPE --task_type=$TASK_TYPE --fold=$FOLD --seed=$SEED


# inference single fold
SEED=10
ITERATION=26000
CUDA_VISIBLE_DEVICES=$GPU_ID python ${WORKSPACE}main.py inference --workspace=$WORKSPACE --feature_dir=$FEATURE_DIR --feature_type=$FEATURE_TYPE --audio_type=$AUDIO_TYPE --task_type=$TASK_TYPE --fold=$FOLD --iteration=$ITERATION --seed=$SEED


# inference all folds
SEED=10
python ${WORKSPACE}main.py inference_all --workspace=$WORKSPACE --audio_type=$AUDIO_TYPE --task_type=$TASK_TYPE --seed=$SEED


# visualization
python utils/my_pred_visualization.py