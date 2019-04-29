import logging
import os
import pdb
from timeit import default_timer as timer

import h5py
import numpy as np

from utils.utilities import (doa_labels, doa_to_ix, event_labels, lb_to_ix,
                             test_split_dict, train_splits_dict,
                             validation_split_dict)


class DataGenerator(object):

    def __init__(self, args, hdf5_dir, logging=logging):
        """

        Inputs:
            args: all parameters
            hdf5_fn: str, path of hdf5 data
            logging: logging file
        """

        # Parameters
        self.fs = args.fs
        self.nfft = args.nfft
        self.hopsize = args.hopsize
        self.mel_bins = args.mel_bins
        self.chunklen = args.chunklen
        self.hopframes = args.hopframes

        self.batch_size = args.batch_size

        self.train_random_state = np.random.RandomState(args.seed)
        self.test_random_state = np.random.RandomState(args.seed)

        self.class_num = len(event_labels)

        train_splits = train_splits_dict[args.fold]
        validation_splits = validation_split_dict[args.fold]
        test_splits = test_split_dict[args.fold]

        hdf5_dev_dir = os.path.join(hdf5_dir + '_dev/')
        hdf5_fns = sorted(os.listdir(hdf5_dev_dir))
        
        self.train_hdf5_fn = [fn for fn in hdf5_fns if int(fn[5]) in train_splits]
        self.validation_hdf5_fn = [fn for fn in hdf5_fns if int(fn[5]) in validation_splits]
        self.test_hdf5_fn = [fn for fn in hdf5_fns if int(fn[5]) in test_splits]

        # Load the segmented data
        load_begin_time = timer()

        # Train data
        pointer = 0
        self.train_features_list = []
        self.train_fn_list = []
        self.train_target_events_list = []
        self.train_target_doas_list = []
        self.train_target_dists_list = []
        self.train_segmented_indexes = []
        
        for hdf5_fn in self.train_hdf5_fn:
            
            fn = hdf5_fn.split('.')[0]
            hdf5_path = os.path.join(hdf5_dev_dir, hdf5_fn)
            feature, target_event, target_doa, target_dist = \
                self.load_hdf5(hdf5_path)

            train_index = []
            # segment, keep only indexes
            frame_num = feature.shape[1]
            if frame_num > self.chunklen:
                train_index = np.arange(pointer, pointer+frame_num-self.chunklen+1, self.hopframes).tolist()
                if (frame_num - self.chunklen) % self.hopframes != 0:
                    train_index.append(pointer+frame_num-self.chunklen)
            elif frame_num < self.chunklen:
                feature = np.concatenate(
                    (feature, \
                        -100*np.ones((feature.shape[0],self.chunklen-frame_num,feature.shape[-1]))), axis=1)
                target_event = np.concatenate(
                    (target_event, \
                        -100*np.ones((self.chunklen-frame_num,target_event.shape[-1]))), axis=0)
                target_doa = np.concatenate(
                    (target_doa, \
                        -100*np.ones((self.chunklen-frame_num,target_doa.shape[-1]))), axis=0) 
                target_dist = np.concatenate(
                    (target_dist, \
                        -100*np.ones((self.chunklen-frame_num,target_dist.shape[-1]))), axis=0)
                train_index.append(pointer)
            elif frame_num == self.chunklen:
                train_index.append(pointer)
            pointer += frame_num

            self.train_features_list.append(feature)
            self.train_fn_list.append(fn)
            self.train_target_events_list.append(target_event)
            self.train_target_doas_list.append(target_doa)
            self.train_target_dists_list.append(target_dist)
            self.train_segmented_indexes.append(train_index)

        self.train_features = np.concatenate(self.train_features_list, axis=1)
        self.train_target_events = np.concatenate(self.train_target_events_list, axis=0)
        self.train_target_doas = np.concatenate(self.train_target_doas_list, axis=0)
        self.train_target_dists = np.concatenate(self.train_target_dists_list, axis=0)
        self.train_segmented_indexes = np.concatenate(self.train_segmented_indexes, axis=0)

        # Validation data
        self.validation_features_list = []
        self.validation_fn_list = []
        self.validation_target_events_list = []
        self.validation_target_doas_list = []
        self.validation_target_dists_list = []
        for hdf5_fn in self.validation_hdf5_fn:
            
            fn = hdf5_fn.split('.')[0]
            hdf5_path = os.path.join(hdf5_dev_dir, hdf5_fn)
            feature, target_event, target_doa, target_dist = \
                self.load_hdf5(hdf5_path)
            
            self.validation_features_list.append(feature)
            self.validation_fn_list.append(fn)
            self.validation_target_events_list.append(target_event)
            self.validation_target_doas_list.append(target_doa)
            self.validation_target_dists_list.append(target_dist)

        # Test data
        self.test_features_list = []
        self.test_fn_list = []
        self.test_target_events_list = []
        self.test_target_doas_list = []
        self.test_target_dists_list = []
        for hdf5_fn in self.test_hdf5_fn:
            
            fn = hdf5_fn.split('.')[0]
            hdf5_path = os.path.join(hdf5_dev_dir, hdf5_fn)
            feature, target_event, target_doa, target_dist = \
                self.load_hdf5(hdf5_path)
            
            self.test_features_list.append(feature)
            self.test_fn_list.append(fn)
            self.test_target_events_list.append(target_event)
            self.test_target_doas_list.append(target_doa)
            self.test_target_dists_list.append(target_dist)

        # Scalar
        scalar_path = os.path.join(hdf5_dir + '_scalar.h5')
        with h5py.File(scalar_path, 'r') as hf_scalar:
            self.mean = hf_scalar['mean'][:]
            self.std = hf_scalar['std'][:]

        load_time = timer() - load_begin_time
        logging.info('Loading training data time: {:.3f} s.\n'.format(load_time))
        logging.info('Training audios number: {}\n'.format(len(self.train_segmented_indexes)))
        logging.info('Cross-Validation audios number: {}\n'.format(len(self.validation_fn_list)))
        logging.info('Testing audios number: {}\n'.format(len(self.test_fn_list)))        

        self.epoch_size = np.ceil(len(self.train_segmented_indexes)/self.batch_size)
        
    def load_hdf5(self, hdf5_path):
        '''
        Load hdf5. 
        
        Args:
          hdf5_path: string
          
        Returns:
          feature: (channel_num, frame_num, freq_bins)
          target_event: (frame_num, class_num)
          target_doa: (frame_num, 2*class_num) for 'regr' | (frame_num, class_num, ele_num*azi_num=324) for 'clas'
          target_dist: (frame_num, class_num) for 'regr' | (frame_num, class_num, 2) for 'clas'
        '''

        with h5py.File(hdf5_path, 'r') as hf:
            feature = hf['feature'][:]
            event = [e.decode() for e in hf['target']['event'][:]]
            start_time = hf['target']['start_time'][:]
            end_time = hf['target']['end_time'][:]
            elevation = hf['target']['elevation'][:]
            azimuth = hf['target']['azimuth'][:]   
            distance = hf['target']['distance'][:]            
        
        frame_num = feature.shape[1]
        target_event = np.zeros((frame_num, self.class_num))
        target_ele = np.zeros((frame_num, self.class_num))
        target_azi = np.zeros((frame_num, self.class_num))
        target_dist = np.zeros((frame_num, self.class_num))
        
        for n in range(len(event)):
            start_idx = np.int(np.round(start_time[n] * self.fs//self.hopsize)) ##### consider it further about this round!!!
            end_idx = np.int(np.round(end_time[n] * self.fs//self.hopsize))
            class_idx = lb_to_ix[event[n]]
            target_event[start_idx:end_idx, class_idx] = 1.0
            target_ele[start_idx:end_idx, class_idx] = elevation[n] * np.pi / 180.0
            target_azi[start_idx:end_idx, class_idx] = azimuth[n] * np.pi / 180.0
            target_dist[start_idx:end_idx, class_idx] = distance[n] * 1.0

        target_doa = np.concatenate((target_azi, target_ele), axis=-1)

        return feature, target_event, target_doa, target_dist

    def transform(self, x):
        """

        Use the calculated scalar to transform data.
        """

        return (x - self.mean) / self.std

    def generate_train(self):
        """Generate batch data for training.

        Returns:
          batch_x: (batch_size, mic_channels, seq_len, freq_bins)
          batch_y_dict: {'events': target_events, 'elevation': target_elevation,
                         'azimuth': target_azimuth, 'distance': target_distance}
        """
        len_x = len(self.train_segmented_indexes)
        indexes = np.array(self.train_segmented_indexes)

        self.train_random_state.shuffle(indexes)

        pointer = 0

        while True:
            
            if pointer >= len_x:
                pointer = 0
                self.train_random_state.shuffle(indexes)
            
            if pointer + self.batch_size <= len_x:
                batch_indexes = indexes[pointer: pointer + self.batch_size]
            else:
                batch_indexes = np.hstack((indexes[pointer: ], indexes[: self.batch_size - (len_x - pointer)]))

            pointer += self.batch_size

            data_idxes = batch_indexes[:, None] + np.arange(self.chunklen)
            batch_x = self.train_features[:, data_idxes]
            batch_x = batch_x.transpose(1, 0, 2, 3)
            batch_x = self.transform(batch_x)
            # batch_x = batch_x[:,:4]
            
            batch_y_dict = {'events': self.train_target_events[data_idxes],
                            'doas': self.train_target_doas[data_idxes],
                            'distances': self.train_target_dists[data_idxes]}
            
            yield batch_x, batch_y_dict

    def generate_test(self, data_type, max_audio_num=None):
        """
        Generate test data for (train, cross validation and test). 

        Args:
          data_type: 'train' | 'validation' | 'test'
          max_audio_num: int, maximum iteration for validation

        Returns:
          batch_x: (batch_size, mic_channels, seq_len, freq_bins)
          batch_y: {'names': names,
                    'events': target_events, 'elevation': target_elevation,
                    'azimuth': target_azimuth, 'distance': target_distance}
        """         

        if data_type == 'train':
            features = self.train_features_list
            fn = self.train_fn_list
            target_events = self.train_target_events_list
            target_doas = self.train_target_doas_list
            target_dists= self.train_target_dists_list
        elif data_type == 'valid':
            features = self.validation_features_list
            fn = self.validation_fn_list
            target_events = self.validation_target_events_list
            target_doas = self.validation_target_doas_list
            target_dists = self.validation_target_dists_list
        elif data_type == 'test':
            features = self.test_features_list
            fn = self.test_fn_list
            target_events = self.test_target_events_list
            target_doas = self.test_target_doas_list
            target_dists = self.test_target_dists_list                    
        else:
            raise Exception('Incorrect data type!')

        len_x = len(features)
        indexes = np.arange(len_x)

        if max_audio_num:
            self.test_random_state.shuffle(indexes)

        for n, idx in enumerate(indexes):
            
            if n == max_audio_num:
                break        

            batch_x = features[idx]
            batch_x = batch_x[None,:,:,:]
            batch_x = self.transform(batch_x)
            # batch_x = batch_x[:,:4]            

            batch_fn = fn[idx]

            batch_y_dict = {'events': target_events[idx],
                            'doas': target_doas[idx],
                            'distances': target_dists[idx]}

            yield batch_x, batch_y_dict, batch_fn

    def generate_train_condi(self, conditional=False):
        """
        Generate batch data for training.
        Conditional doa

        Input:
            conditional: generate conditional data, feature concatenate with event labels

        Returns:
            batch_x: (batch_size, mic_channels, seq_len, freq_bins)
            batch_y_dict: {'events': target_events, 'elevation': target_elevation,
                            'azimuth': target_azimuth, 'distance': target_distance}
        """
        len_x = len(self.train_segmented_indexes)
        indexes = np.array(self.train_segmented_indexes)

        self.train_random_state.shuffle(indexes)

        pointer = 0

        while True:
            
            if pointer >= len_x:
                pointer = 0
                self.train_random_state.shuffle(indexes)
            
            if pointer + self.batch_size <= len_x:
                batch_indexes = indexes[pointer: pointer + self.batch_size]
            else:
                batch_indexes = np.hstack((indexes[pointer: ], indexes[: self.batch_size - (len_x - pointer)]))

            pointer += self.batch_size

            data_idxes = batch_indexes[:, None] + np.arange(self.chunklen)
            batch_x = self.train_features[:, data_idxes]
            batch_x = batch_x.transpose(1, 0, 2, 3)
            batch_x = self.transform(batch_x)
            '''(batch_size, channel_num=feature_map, time, frequency)'''

            if conditional:
                batch_event = self.train_target_events[data_idxes]
                '''(batch_size, time, class_num=feature_map)'''
                batch_event = np.expand_dims(batch_event, axis=1)
                batch_event = np.repeat(batch_event, batch_x.shape[-1], axis=1)
                '''(batch_size, frequency(expanded), time, class_num=feature_map)'''
                batch_event = batch_event.transpose(0, 3, 2, 1)
                '''(batch_size, class_num=feature_map, time, frequency(expanded))'''
                batch_x = np.concatenate((batch_x, batch_event), axis=1)
                '''(batch_size, channel_num+class_num=feature_map, time, frequency)'''

            batch_y_dict = {'events': self.train_target_events[data_idxes],
                            'doas': self.train_target_doas[data_idxes],
                            'distances': self.train_target_dists[data_idxes]}
            
            yield batch_x, batch_y_dict

    def generate_test_condi(self, data_type, max_audio_num=None, conditional=False):
        """
        Generate test data for (train, cross validation and test). 
        Conditional doa

        Args:
            data_type: 'train' | 'validation' | 'test'
            max_audio_num: int, maximum iteration for validation
            conditional: generate conditional data, feature concatenate with event labels

        Returns:
            batch_x: (batch_size, mic_channels, seq_len, freq_bins)
            batch_y: {'names': names,
                        'events': target_events, 'elevation': target_elevation,
                        'azimuth': target_azimuth, 'distance': target_distance}
        """         

        if data_type == 'train':
            features = self.train_features_list
            fn = self.train_fn_list
            target_events = self.train_target_events_list
            target_doas = self.train_target_doas_list
            target_dists= self.train_target_dists_list
        elif data_type == 'valid':
            features = self.validation_features_list
            fn = self.validation_fn_list
            target_events = self.validation_target_events_list
            target_doas = self.validation_target_doas_list
            target_dists = self.validation_target_dists_list
        elif data_type == 'test':
            features = self.test_features_list
            fn = self.test_fn_list
            target_events = self.test_target_events_list
            target_doas = self.test_target_doas_list
            target_dists = self.test_target_dists_list                    
        else:
            raise Exception('Incorrect data type!')

        len_x = len(features)
        indexes = np.arange(len_x)

        if max_audio_num:
            self.test_random_state.shuffle(indexes)

        for n, idx in enumerate(indexes):
            
            if n == max_audio_num:
                break        

            batch_x = features[idx]
            batch_x = batch_x[None,:,:,:]
            batch_x = self.transform(batch_x)
            
            if conditional:
                batch_event = target_events[idx]
                '''(time, class_num=feature_map)'''
                batch_event = np.expand_dims(batch_event, axis=0)
                batch_event = np.expand_dims(batch_event, axis=0)
                batch_event = np.repeat(batch_event, batch_x.shape[-1], axis=1)
                '''(batch_size, frequency(expanded), time, class_num=feature_map)'''
                batch_event = batch_event.transpose(0, 3, 2, 1)
                '''(batch_size, class_num=feature_map, time, frequency(expanded))'''
                batch_x = np.concatenate((batch_x, batch_event), axis=1)
                '''(batch_size, channel_num+class_num=feature_map, time, frequency)'''

            batch_fn = fn[idx]

            batch_y_dict = {'events': target_events[idx],
                            'doas': target_doas[idx],
                            'distances': target_dists[idx]}

            yield batch_x, batch_y_dict, batch_fn
