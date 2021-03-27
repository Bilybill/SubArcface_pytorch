from torch.utils.data.sampler import Sampler
import numpy as np


class GivenIterationSampler(Sampler):
    def __init__(self, dataset, total_iter, batch_size, last_iter=-1):
        self.dataset = dataset
        self.total_iter = total_iter
        self.batch_size = batch_size
        self.last_iter = last_iter
        self.total_size = self.total_iter * self.batch_size
        self.indices = self.gen_new_list()

    def __iter__(self):
        return iter(self.indices)

    def gen_new_list(self):
        np.random.seed(233)
        indices = np.arange(len(self.dataset))
        num_repeat = self.total_size // indices.shape[0] + 1
        indices = np.tile(indices, num_repeat)
        indices = indices[: self.total_size]
        np.random.shuffle(indices)
        return indices

    def __len__(self):
        return self.total_size - (self.last_iter + 1) * self.batch_size
