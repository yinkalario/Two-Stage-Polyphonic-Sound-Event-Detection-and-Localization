#!/bin/bash

# Data directory
DATASET_DIR='/vol/vssp/AP_datasets/audio/dcase2019/task3/dataset_root/'

# Feature directory
FEATURE_DIR='/vol/vssp/msos/YinC/workspace/Dataset_Features/DCASE2019/task3/'

# Workspace
WORKSPACE='/vol/vssp/msos/YinC/workspace/DCASE2019/task3/mycode_v11/'
cd $WORKSPACE

FEATURE_TYPE='logmelgcc'
AUDIO_TYPE='foa'

# TASK_TYPE: 'sed_only' | 'doa_only' | 'two_staged_eval' | 'seld'
TASK_TYPE='sed_only'

SEED=10
ITERATION=26000

GPU_ID=3

############ Development Evaluation ############

# inference single fold
for FOLD in {1..4}
    do
    echo $'\nFold: '$FOLD
    CUDA_VISIBLE_DEVICES=$GPU_ID python main.py inference --workspace=$WORKSPACE --feature_dir=$FEATURE_DIR --feature_type=$FEATURE_TYPE --audio_type=$AUDIO_TYPE --task_type=$TASK_TYPE --fold=$FOLD --iteration=$ITERATION --seed=$SEED
done

# inference all folds
python main.py inference_all --workspace=$WORKSPACE --audio_type=$AUDIO_TYPE --task_type=$TASK_TYPE --seed=$SEED
