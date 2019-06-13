#!/bin/bash

# Data directory
DATASET_DIR='/vol/vssp/AP_datasets/audio/dcase2019/task3/dataset_root/'

# Feature directory
FEATURE_DIR='/vol/vssp/msos/YinC/workspace/Dataset_Features/DCASE2019/task3/'

# Workspace
WORKSPACE='/vol/vssp/msos/YinC/workspace/DCASE2019/task3/'
cd $WORKSPACE

FEATURE_TYPE='logmelgcc'
AUDIO_TYPE='mic'
SEED=10

# TASK_TYPE: 'sed_only' | 'doa_only' | 'two_staged_eval' | 'seld'
TASK_TYPE='two_staged_eval'

# which model to use
ITERATION=50000

# GPU number
GPU_ID=0

############ Development Evaluation ############

# inference single fold
for FOLD in {1..4}
    do
    echo $'\nFold: '$FOLD
    CUDA_VISIBLE_DEVICES=$GPU_ID python ${WORKSPACE}main.py inference --workspace=$WORKSPACE --feature_dir=$FEATURE_DIR --dataset_dir=$DATASET_DIR --feature_type=$FEATURE_TYPE --audio_type=$AUDIO_TYPE --task_type=$TASK_TYPE --fold=$FOLD --iteration=$ITERATION --seed=$SEED
done

# inference all folds
python ${WORKSPACE}main.py inference_all --workspace=$WORKSPACE --dataset_dir=$DATASET_DIR --audio_type=$AUDIO_TYPE --task_type=$TASK_TYPE --seed=$SEED
