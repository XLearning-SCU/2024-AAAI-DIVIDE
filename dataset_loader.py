import os.path
import torch
from torch.utils.data import ConcatDataset, Dataset
import scipy.io as sio
from scipy import sparse

import sklearn.preprocessing as skp


def load_mat(args):
    data_X = []
    label_y = None

    if args.dataset == 'Scene15':
        mat = sio.loadmat(os.path.join(args.data_path, 'Scene_15.mat'))
        X = mat['X'][0]
        data_X.append(X[0].astype('float32'))
        data_X.append(X[1].astype('float32'))
        label_y = np.squeeze(mat['Y'])

    elif args.dataset == 'LandUse21':
        mat = sio.loadmat(os.path.join(args.data_path, 'LandUse_21.mat'))
        data_X.append(sparse.csr_matrix(mat['X'][0, 1]).A)
        data_X.append(sparse.csr_matrix(mat['X'][0, 2]).A)

        label_y = np.squeeze(mat['Y']).astype('int')

    elif args.dataset == 'Reuters':
        mat = sio.loadmat(os.path.join(args.data_path, 'Reuters_dim10.mat'))
        data_X = []  # 18758 samples
        data_X.append(np.vstack((mat['x_train'][0], mat['x_test'][0])))
        data_X.append(np.vstack((mat['x_train'][1], mat['x_test'][1])))
        label_y = np.squeeze(np.hstack((mat['y_train'], mat['y_test'])))

    elif args.dataset == 'Caltech101':
        mat = sio.loadmat(os.path.join(args.data_path, '2view-caltech101-8677sample.mat'))
        X = mat['X'][0]
        data_X.append(X[0].T)
        data_X.append(X[1].T)
        label_y = np.squeeze(mat['gt']) - 1

    else:
        raise 'Unknown Dataset'

    if args.data_norm == 'standard':
        for i in range(args.n_views):
            data_X[i] = skp.scale(data_X[i])
    elif args.data_norm == 'l2-norm':
        for i in range(args.n_views):
            data_X[i] = skp.normalize(data_X[i])
    elif args.data_norm == 'min-max':
        for i in range(args.n_views):
            data_X[i] = skp.minmax_scale(data_X[i])

    args.n_sample = data_X[0].shape[0]
    return data_X, label_y


def load_dataset(args):
    data, targets = load_mat(args)
    dataset = IncompleteMultiviewDataset(args.n_views, data, targets, args.missing_rate)
    return dataset


class MultiviewDataset(torch.utils.data.Dataset):
    def __init__(self, n_views, data_X, label_y):
        super(MultiviewDataset, self).__init__()
        self.n_views = n_views
        self.data = data_X
        self.targets = label_y - np.min(label_y)

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        data = []
        for i in range(self.n_views):
            data.append(torch.tensor(self.data[i][idx].astype('float32')))
        label = torch.tensor(self.targets[idx], dtype=torch.long)
        return idx, data, label


import numpy as np
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder


class IncompleteMultiviewDataset(torch.utils.data.Dataset):
    def __init__(self, n_views, data_X, label_y, missing_rate):
        super(IncompleteMultiviewDataset, self).__init__()
        self.n_views = n_views
        self.data = data_X
        self.targets = label_y - np.min(label_y)

        self.missing_mask = torch.from_numpy(self._get_mask(n_views, self.data[0].shape[0], missing_rate)).bool()

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        data = []
        for i in range(self.n_views):
            data.append(torch.tensor(self.data[i][idx].astype('float32')))
        label = torch.tensor(self.targets[idx], dtype=torch.long)
        mask = self.missing_mask[idx]
        return idx, data, mask, label

    @staticmethod
    def _get_mask(view_num, alldata_len, missing_rate):
        """Randomly generate incomplete data information, simulate partial view data with complete view data
        :param view_num:view number
        :param alldata_len:number of samples
        :param missing_rate:Defined in section 4.1 of the paper
        :return: mask
        """
        full_matrix = np.ones((int(alldata_len * (1 - missing_rate)), view_num))

        alldata_len = alldata_len - int(alldata_len * (1 - missing_rate))
        missing_rate = 0.5
        if alldata_len != 0:
            one_rate = 1.0 - missing_rate
            if one_rate <= (1 / view_num):
                enc = OneHotEncoder()  # n_values=view_num
                view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
                full_matrix = np.concatenate([view_preserve, full_matrix], axis=0)
                choice = np.random.choice(full_matrix.shape[0], size=full_matrix.shape[0], replace=False)
                matrix = full_matrix[choice]
                return matrix
            error = 1
            if one_rate == 1:
                matrix = randint(1, 2, size=(alldata_len, view_num))
                full_matrix = np.concatenate([matrix, full_matrix], axis=0)
                choice = np.random.choice(full_matrix.shape[0], size=full_matrix.shape[0], replace=False)
                matrix = full_matrix[choice]
                return matrix
            while error >= 0.005:
                enc = OneHotEncoder()  # n_values=view_num
                view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
                one_num = view_num * alldata_len * one_rate - alldata_len
                ratio = one_num / (view_num * alldata_len)
                matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int)
                a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int))
                one_num_iter = one_num / (1 - a / one_num)
                ratio = one_num_iter / (view_num * alldata_len)
                matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int)
                matrix = ((matrix_iter + view_preserve) > 0).astype(np.int)
                ratio = np.sum(matrix) / (view_num * alldata_len)
                error = abs(one_rate - ratio)
            full_matrix = np.concatenate([matrix, full_matrix], axis=0)

        choice = np.random.choice(full_matrix.shape[0], size=full_matrix.shape[0], replace=False)
        matrix = full_matrix[choice]
        return matrix


class IncompleteDatasetSampler:
    def __init__(self, dataset: Dataset, seed: int = 0, drop_last: bool = False) -> None:
        self.dataset = dataset
        self.epoch = 0
        self.drop_last = drop_last
        self.seed = seed
        self.compelte_idx = torch.where(self.dataset.missing_mask.sum(dim=1) == self.dataset.n_views)[0]
        self.num_samples = self.compelte_idx.shape[0]

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = torch.randperm(self.num_samples, generator=g).tolist()

        indices = self.compelte_idx[indices].tolist()

        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        self.epoch = epoch


class DatasetWithIndex(Dataset):
    def __getitem__(self, idx):
        img, label = super(DatasetWithIndex, self).__getitem__(idx)
        return idx, img, label
