import pickle
import torch.utils.data as data
from .tools import *


class Feeder(data.Dataset):

    def __init__(self,
                 data_path,
                 label_path,
                 num_frame_path=None,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 debug=False,
                 mmap=True,
                 label_type=1):
        self.debug = debug
        self.data_path = data_path
        self.num_frame_path = num_frame_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size
        self.label_type = label_type

        self.load_data(mmap)

    def load_data(self, mmap):
        # data: N C V T M

        # load label
        if self.label_type == 1:
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f)
        else:
            self.label = np.load(self.label_path, mmap_mode='r')

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        if self.num_frame_path != None:
            self.number_of_frames = np.load(self.num_frame_path)

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        # processing
        if self.random_choose:
            data_numpy = random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = random_move(data_numpy)

        if self.num_frame_path != None:
            number_of_frames = self.number_of_frames[index]
            return data_numpy, label, number_of_frames
        else:
            return data_numpy, label

class Feeder2(data.Dataset):

    def __init__(self,
                 data_path):
        self.data_path = data_path

        self.load_data()

    def load_data(self):
        # data: N C V T M

        # load data
        self.data = np.load(self.data_path, mmap_mode='r')
        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])

        return data_numpy