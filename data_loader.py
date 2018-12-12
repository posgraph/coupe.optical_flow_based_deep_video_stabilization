#from utils import *
import collections
import numpy as np
import tensorlayer as tl
import cv2
import os
from threading import Thread
from threading import Lock
from datetime import datetime

class Data_Loader:
    def __init__(self, skip_length, stab_path, unstab_path, height, width, batch_size, is_train, thread_num = 3):

        self.is_train = is_train
        self.thread_num = thread_num

        self.num_partition = 1
        self.skip_length = np.array(skip_length)

        self.stab_folder_path_list, self.stab_file_path_list, self.num_files = self._load_file_list(stab_path)
        self.unstab_folder_path_list, self.unstab_file_path_list, _ = self._load_file_list(unstab_path)

        self.h = height
        self.w = width
        self.batch_size = batch_size

    def init_data_loader(self, inputs):
        self.sample_num = 9

        self.idx_video = []
        self.idx_frame = []
        self.init_idx()

        self.num_itr = int(np.ceil(len(sum(self.idx_frame, [])) / self.batch_size))

        self.lock = Lock()
        self.is_end = False

        ### THREAD HOLDERS ###
        self.net_placeholder_names = list(inputs.keys())
        self.net_inputs = inputs

        self.threads = [None] * self.thread_num
        self.threads_unused = [None] * self.thread_num
        self.feed_dict_holder = self._set_feed_dict_holder(self.net_placeholder_names, self.thread_num)
        self._init_data_thread()
        #self._print_log()

    def init_idx(self):
        self.idx_video = []
        self.idx_frame = []
        for i in range(len(self.stab_file_path_list)):
            total_frames = len(self.stab_file_path_list[i])

            #self.idx_frame.append(list(range(0, total_frames - ((self.sample_num - 1) * self.skip_length + 1) - (self.num_partition - 1))))
            self.idx_frame.append(list(range(0, total_frames - (self.skip_length[-1] - self.skip_length[0] + 1) - (self.num_partition - 1))))
            for j in np.arange(self.sample_num - 1):
                self.idx_frame[i].append(0)

            self.idx_video.append(i)

        self.is_end = False

    def get_batch(self, threads_unused, thread_idx):
        assert(self.net_placeholder_names is not None)
        #tl.logging.debug('\tthread[%s] > get_batch start [%s]' % (str(thread_idx), str(datetime.now())))

        ## random sample frame indexes
        self.lock.acquire()
        #tl.logging.debug('\t\tthread[%s] > acquired lock [%s]' % (str(thread_idx), str(datetime.now())))

        if self.is_end:
            #tl.logging.debug('\t\tthread[%s] > releasing lock 1 [%s]' % (str(thread_idx), str(datetime.now())))
            self.lock.release()
            return

        video_idxes = []
        frame_offsets = []

        actual_batch = 0
        for i in range(0, self.batch_size):
            if i == 0 and len(self.idx_video) == 0:
                self.is_end = True
                #tl.logging.debug('\t\tthread[%s] > releasing lock 2 [%s]' % (str(thread_idx), str(datetime.now())))
                self.lock.release()
                return
            elif i > 0 and len(self.idx_video) == 0:
                break

            else:
                if self.is_train:
                    idx_x = np.random.randint(len(self.idx_video))
                    video_idx = self.idx_video[idx_x]
                    idx_y = np.random.randint(len(self.idx_frame[video_idx]))
                else:
                    idx_x = 0
                    idx_y = 0
                    video_idx = self.idx_video[idx_x]

            frame_offset = self.idx_frame[video_idx][idx_y]
            video_idxes.append(video_idx)
            frame_offsets.append(frame_offset)
            self._update_idx(idx_x, idx_y)
            actual_batch += 1

        #tl.logging.debug('\t\tthread[%s] > releasing lock 4 [%s]' % (str(thread_idx), str(datetime.now())))
        self.lock.release()
        threads_unused[thread_idx] = True

        ## init thread lists
        data_holder = self._set_data_holder(self.net_placeholder_names, actual_batch)

        ## start thread
        threads = [None] * actual_batch
        for batch_idx in range(actual_batch):
            video_idx = video_idxes[batch_idx]
            frame_offset = frame_offsets[batch_idx]
            threads[batch_idx] = Thread(target = self.read_dataset, args = (data_holder, batch_idx, video_idx, frame_offset))
            threads[batch_idx].start()

        for batch_idx in range(actual_batch):
            threads[batch_idx].join()

        for (key, val) in data_holder.items():
            data_holder[key] = np.concatenate(data_holder[key][0 : actual_batch], axis = 0)

        for holder_name in self.net_placeholder_names:
            self.feed_dict_holder[holder_name][thread_idx] = data_holder[holder_name]

        #tl.logging.debug('\tthread[%s] > get_batch done [%s]' % (str(thread_idx), str(datetime.now())))

    def read_dataset(self, data_holder, batch_idx, video_idx, frame_offset):
        #sampled_frame_idx = np.arange(frame_offset, frame_offset + self.sample_num * self.skip_length, self.skip_length)
        sampled_frame_idx = frame_offset + self.skip_length

        patches_temp_S = [None] * (len(sampled_frame_idx) - 1)

        threads = [None] * len(sampled_frame_idx)
        for frame_idx in range(len(sampled_frame_idx)):

            is_last = True if frame_idx == len(sampled_frame_idx) - 1 else False

            sampled_idx = sampled_frame_idx[frame_idx]
            threads[frame_idx] = Thread(target = self.read_frame_data, args = (data_holder, batch_idx, video_idx, frame_idx, sampled_idx, patches_temp_S, is_last))

            threads[frame_idx].start()

        for frame_idx in range(len(sampled_frame_idx)):
            threads[frame_idx].join()

        patches_temp_S = np.concatenate(patches_temp_S[0: len(patches_temp_S)], axis = 3)
        data_holder['stab_image'][batch_idx] = patches_temp_S

    def read_frame_data(self, data_holder, batch_idx, video_idx, frame_idx, sampled_idx, patches_temp_S, is_last):
        # get stab/unstab file path
        stab_file_path = self.stab_file_path_list[video_idx]
        unstab_file_path = self.unstab_file_path_list[video_idx]

        stab_frame = self._read_frame(stab_file_path[sampled_idx])
        unstab_frame = self._read_frame(unstab_file_path[sampled_idx])

        assert(self._get_folder_name(unstab_file_path[sampled_idx]) == self._get_folder_name(stab_file_path[sampled_idx]))
        assert(self._get_base_name(unstab_file_path[sampled_idx]) == self._get_base_name(stab_file_path[sampled_idx]))

        if is_last is False:
            patches_temp_S[frame_idx] = stab_frame
        else:
            data_holder['gtstab_image'][batch_idx] = stab_frame
            data_holder['unstab_image'][batch_idx] = unstab_frame

        # print('batch_idx: ', str(batch_idx), ' video_idx: ', str(video_idx), ' frame_idx: ', str(frame_idx), ' frame: ', stab_file_path[sampled_idx_t])

    def _update_idx(self, idx_x, idx_y):
        video_idx = self.idx_video[idx_x]
        del(self.idx_frame[video_idx][idx_y])

        if len(self.idx_frame[video_idx]) == 0:
            del(self.idx_video[idx_x])
            # if len(self.idx_video) != 0:
            #     self.video_name = os.path.basename(self.stab_file_path_list[self.idx_video[0]])

    def _load_file_list(self, root_path):
        folder_paths = []
        file_names = []
        num_files = 0
        for root, dirnames, filenames in os.walk(root_path):
            if len(dirnames) == 0:
                folder_paths.append(root)
                for i in np.arange(len(filenames)):
                    filenames[i] = os.path.join(root, filenames[i])
                file_names.append(np.array(sorted(filenames)))
                num_files += len(filenames)

        folder_paths = np.array(folder_paths)
        file_names = np.array(file_names)

        sort_idx = np.argsort(folder_paths)
        folder_paths = folder_paths[sort_idx]
        file_names = file_names[sort_idx]

        return np.squeeze(folder_paths), np.squeeze(file_names), np.squeeze(num_files)

    def _read_frame(self, path):
        frame = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) / 255.
        frame = cv2.resize(frame, (self.w, self.h))
        return np.reshape(frame, [1, self.h, self.w, 3])


    def _get_base_name(self, path):
        return os.path.basename(path.split('.')[0])

    def _get_folder_name(self, path):
        path = os.path.dirname(path)
        return path.split(os.sep)[-1]

    def _set_feed_dict_holder(self, holder_names, thread_num):
        feed_dict_holder = collections.OrderedDict()
        for holder_name in holder_names:
            feed_dict_holder[holder_name] = [None] * thread_num

        return feed_dict_holder

    def _set_data_holder(self, net_placeholder_names, batch_num):
        data_holder = collections.OrderedDict()
        for holder_name in net_placeholder_names:
            data_holder[holder_name] = [None] * batch_num

        return data_holder

    def _init_data_thread(self):
        self.init_idx()
        #tl.logging.debug('INIT_THREAD [%s]' % str(datetime.now()))
        for thread_idx in range(0, self.thread_num):
            self.threads[thread_idx] = Thread(target = self.get_batch, args = (self.threads_unused, thread_idx))
            self.threads_unused[thread_idx] = False
            self.threads[thread_idx].start()

        #tl.logging.debug('INIT_THREAD DONE [%s]' % str(datetime.now()))

    def feed_the_network(self):
        thread_idx, is_end = self._get_thread_idx()
        #tl.logging.debug('THREAD[%s] > FEED_THE_NETWORK [%s]' % (str(thread_idx), str(datetime.now())))
        is_not_batchsize = False
        if is_end:
            return None, is_end, is_not_batchsize
        

        feed_dict = collections.OrderedDict()
        for (key, val) in self.net_inputs.items():
            feed_dict[val] = self.feed_dict_holder[key][thread_idx]
            if self.feed_dict_holder[key][thread_idx].shape[0] != self.batch_size:
                is_not_batchsize = True 

        #tl.logging.debug('THREAD[%s] > FEED_THE_NETWORK DONE [%s]' % (str(thread_idx), str(datetime.now())))
        return feed_dict, is_end, is_not_batchsize

    def _get_thread_idx(self):
        for thread_idx in np.arange(self.thread_num):
            if self.threads[thread_idx].is_alive() == False and self.threads_unused[thread_idx] == False:
                    self.threads[thread_idx] = Thread(target = self.get_batch, args = (self.threads_unused, thread_idx))
                    self.threads[thread_idx].start()

        while True:
            is_unused_left = False
            for thread_idx in np.arange(self.thread_num):
                if self.threads_unused[thread_idx]:
                    is_unused_left = True
                    if self.threads[thread_idx].is_alive() == False:
                        self.threads_unused[thread_idx] = False
                        return thread_idx, False

            if is_unused_left == False and self.is_end:
                self._init_data_thread()
                return None, True

    def _print_log(self):
        print('stab_folder_path_list')
        print(len(self.stab_folder_path_list))

        print('stab_file_path_list')
        total_file_num = 0
        for file_path in self.stab_file_path_list:
            total_file_num += len(file_path)
        print(total_file_num)

        print('unstab_file_path_list')
        total_file_num = 0
        for file_path in self.unstab_file_path_list:
            total_file_num += len(file_path)
        print(total_file_num)

        print('num itr per epoch')
        print(self.num_itr)
