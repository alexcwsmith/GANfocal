import numpy as np
import os
from skimage import io
from scipy.ndimage import gaussian_filter
from skimage.util import view_as_windows

class Train_dataset(object):
    def __init__(self, data_path, zdepth, height, width, batch_size, train_portion=0.8):
        self.data_path = data_path
        self.zdepth = zdepth
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.file_list = os.listdir(self.data_path)
        self.train_size = int(round(len(self.file_list)*train_portion))
        self.file_list = self.file_list[:self.train_size]


    def retrieveData(self, iteration):
        data_batch = self.file_list[iteration * self.batch_size:self.batch_size + (iteration * self.batch_size)]
        data = np.empty([self.batch_size, self.zdepth, self.height, self.width]) 
        i = 0
        for image in data_batch:
            filename = os.path.join(self.data_path, image)
            im = io.imread(filename)
            im = im[:self.zdepth,:self.height,:self.width]
            data[i] = im
            i = i+1
        return data
    
    def blurData(self, iteration):
        data_batch = self.file_list[iteration * self.batch_size:self.batch_size + (iteration * self.batch_size)]
        data_blur = np.empty([self.batch_size, self.zdepth, self.height, self.width]) 
        i = 0
        for image in data_batch:
            filename = os.path.join(self.data_path, image)
            im = io.imread(filename)
            im = im[:self.zdepth,:self.height,:self.width]
            im = gaussian_filter(im, sigma=1.2, order=0)
            data_blur[i] = im
            i = i+1
        return data_blur
    
    def dataToTensor(self, iteration, blur=False):
        if blur:
            data_true = self.blurData(iteration)
        else:
            data_true = self.retrieveData(iteration)
        data_tensor = np.empty(
            [self.batch_size, self.zdepth, self.height,
             self.width, 1])
        i=0
        for image in data_true:
            patch = view_as_windows(image, window_shape=(
                ((self.zdepth), (self.height), self.width)),
                                    step=(self.zdepth, self.height,
                                          self.width))
           
            for d in range(patch.shape[0]):
                for v in range(patch.shape[1]):
                    for h in range(patch.shape[2]):
                        dt = patch[d, v, h, :]
                        dt = dt[:, np.newaxis]
                        dt = dt.transpose(0,2,3,1)
                        data_tensor[i] = dt
                        i = i+1
        return data_tensor
    
class Test_dataset(object):
    def __init__(self, data_path, zdepth, height, width, batch_size, test_portion=0.2):
        self.data_path = data_path
        self.zdepth = zdepth
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.file_list = os.listdir(self.data_path)
        self.test_size = int(round(len(self.file_list)*test_portion))
        self.file_list = self.file_list[-self.test_size:]

    def shuffle():
        return(0.8)

    def retrieveData(self, iteration):
        data_batch = self.file_list[iteration * self.batch_size:self.batch_size + (iteration * self.batch_size)]
        data = np.empty([self.batch_size, self.zdepth, self.height, self.width]) 
        i = 0
        for image in data_batch:
            filename = os.path.join(self.data_path, image)
            im = io.imread(filename)
            im = im[:self.zdepth,:self.height,:self.width]
            data[i] = im
            i = i+1
        return data
    
    def blurData(self, iteration):
        data_batch = self.file_list[iteration * self.batch_size:self.batch_size + (iteration * self.batch_size)]
        data_blur = np.empty([self.batch_size, self.zdepth, self.height, self.width]) 
        i = 0
        for image in data_batch:
            filename = os.path.join(self.data_path, image)
            im = io.imread(filename)
            im = im[:self.zdepth,:self.height,:self.width]
            im = gaussian_filter(im, sigma=1.2, order=0)
            data_blur[i] = im
            i = i+1
        return data_blur
    
    def dataToTensor(self, iteration, blur=False):
        if blur:
            data_true = self.blurData(iteration)
        else:
            data_true = self.retrieveData(iteration)
        data_tensor = np.empty(
            [self.batch_size, self.zdepth, self.height,
             self.width, 1])
        i=0
        for image in data_true:
            patch = view_as_windows(image, window_shape=(
                ((self.zdepth), (self.height), self.width)),
                                    step=(self.zdepth, self.height,
                                          self.width))
           
            for d in range(patch.shape[0]):
                for v in range(patch.shape[1]):
                    for h in range(patch.shape[2]):
                        dt = patch[d, v, h, :]
                        dt = dt[:, np.newaxis]
                        dt = dt.transpose(0,2,3,1)
                        data_tensor[i] = dt
                        i = i+1
        return data_tensor


