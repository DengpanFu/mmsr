"""
Modified from torch.utils.data.distributed.DistributedSampler
Support enlarging the dataset for *iteration-oriented* training, for saving time when restart the
dataloader after each epoch
"""
import math
import torch
from torch.utils.data.sampler import Sampler
import torch.distributed as dist
import numpy as np


class DistIterSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, ratio=100):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * ratio / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(self.total_size, generator=g).tolist()

        dsize = len(self.dataset)
        indices = [v % dsize for v in indices]

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistMetaIterSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, batchsize=32, 
        num_scales=30, ratio=1):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batchsize = batchsize
        assert(batchsize % num_replicas == 0)
        self.num_scales = num_scales
        self.ratio = ratio
        self.epoch = 0
        self.bs_per_dev = batchsize // num_replicas
        # self.num_samples = int(math.ceil(len(self.dataset) * ratio / self.num_replicas))
        num_samples = int(math.ceil(len(self.dataset) * ratio) / \
                self.batchsize / self.num_scales) # 26600*300/32/30=8312
        self.total_size = num_samples * self.batchsize * self.num_scales  # 8312*32*30=7979520
        self.num_samples = self.total_size // self.num_replicas  # 7979520/2=3989760



    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        # generate image index
        indices = torch.randperm(self.total_size, generator=g).tolist()  # 7979520
        dsize = len(self.dataset)   # 26600
        indices = [v % dsize for v in indices]
        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas] 
        assert len(indices) == self.num_samples

        # generate scale index
        scales_templet = torch.arange(self.num_scales).repeat(self.bs_per_dev, 1).t() # 30,16
        rand_start = torch.randperm(self.total_size // self.bs_per_dev, generator=g)  # 498720
        scale_index = scales_templet[rand_start % self.num_scales]                    # 498720,16
        # print("rank={:d} ==> scale_index_shape: {}".format(self.rank, scale_index.shape))
        scale_index = scale_index[self.rank:scale_index.shape[0]:self.num_replicas]
        # print(scale_index.shape)
        scale_index = scale_index.reshape(-1).tolist()
        # print("rank={:d} ==> scale_index_len={}".format(self.rank, len(scale_index)))
        assert len(scale_index) == self.num_samples

        return iter(zip(indices, scale_index))

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
