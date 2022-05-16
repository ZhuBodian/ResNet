import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit
import torch


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """

    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate,
                 assigned_val=False, samplers=None):
        """
        @param dataset: torchvision.datasets的实例化类
        @param batch_size:
        @param shuffle:
        @param validation_split: 验证集比例，0~1
        @param num_workers:
        @param collate_fn:
        """
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.assigned_val = assigned_val

        self.batch_idx = 0
        self.n_samples = len(dataset)

        assert (not assigned_val) or (assigned_val and samplers is not None), print('令assigned_val=False，或给定训练集、验证集samples')
        # 如果事先指定好验证集的索引，就不再随机指定了
        if not self.assigned_val:
            self.sampler, self.valid_sampler = self._split_sampler(self.validation_split, self.assigned_val)
        else:
            self.sampler, self.valid_sampler = samplers[0], samplers[1]
            self._split_sampler(self.validation_split, self.assigned_val)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs, pin_memory=torch.cuda.is_available())

    def _split_sampler(self, split, assigned_val):
        """这个就是随机抽样，mnist各类别均匀，随机抽样倒是也行，但是对于不均匀样本还是分层抽样"""
        if split == 0.0:
            return None, None

        """# 原代码是随机抽样，自己改为分层抽样
        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)
        """

        if not assigned_val:
            split = StratifiedShuffleSplit(n_splits=1, test_size=split, random_state=42)
            for train_index, valid_index in split.split(np.array(self.dataset.data), np.array(self.dataset.targets)):
                train_sampler, valid_sampler = SubsetRandomSampler(train_index), SubsetRandomSampler(valid_index)
                self.shuffle = False
                self.n_samples = len(train_index)
                return train_sampler, valid_sampler
        else:
            self.shuffle = False
            self.n_samples = len(self.dataset.samples[0])


    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
