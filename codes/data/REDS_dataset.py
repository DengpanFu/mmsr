'''
REDS dataset
support reading images from lmdb, image folder and memcached
'''
import os.path as osp
import os, random
import pickle
import logging
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
try:
    import mc  # import memcached
except ImportError:
    pass
import matplotlib.pyplot as plt

logger = logging.getLogger('base')

class REDSDataset(data.Dataset):
    '''
    Reading the training REDS dataset
    key example: 000_00000000
    GT: Ground-Truth;
    LQ: Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames
    support reading N LQ frames, N = 1, 3, 5, 7
    '''
    def __init__(self, opt):
        super(REDSDataset, self).__init__()
        self.opt = opt
        # temporal augmentation
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        logger.info('Temporal augmentation interval list: [{}], with random reverse is {}.'.format(
            ','.join(str(x) for x in opt['interval_list']), self.random_reverse))

        self.half_N_frames = opt['N_frames'] // 2
        self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.data_type = self.opt['data_type']
        self.scale = opt['scale']

        #### Load image keys
        if self.data_type == 'lmdb':
            self.paths_GT, self.GT_size_tuple = util.get_image_paths(self.data_type, self.GT_root)
            logger.info('Using lmdb meta info for cache keys.')
            self.paths_GT = [v for v in self.paths_GT if v.split('_')[0] not in \
                            ['000', '011', '015', '020']]
            self.LQ_size_tuple = [self.GT_size_tuple[0], int(self.GT_size_tuple[1] / self.scale), \
                                    int(self.GT_size_tuple[2] / self.scale)]
        else:
            seqs = sorted(os.listdir(self.GT_root))
            self.paths_GT = []
            for seq in seqs:
                if not seq in ['000', '011', '015', '020']:
                    names = os.listdir(osp.join(self.GT_root, seq))
                    self.paths_GT.extend([seq + '_' + x[:-4] for x in names])
        assert self.paths_GT, 'Error: GT path is empty.'

        if self.data_type == 'lmdb':
            self.GT_env, self.LQ_env = None, None

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.GT_root, readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.LQ_env = lmdb.open(self.LQ_root, readonly=True, lock=False, readahead=False,
                                meminit=False)

    def get_neighbor_list(self, center_frame_idx):
        #### determine the neighbor frames
        interval = random.choice(self.interval_list)
        if self.opt['border_mode']:
            direction = 1  # 1: forward; 0: backward
            N_frames = self.opt['N_frames']
            if self.random_reverse and random.random() < 0.5:
                direction = random.choice([0, 1])
            if center_frame_idx + interval * (N_frames - 1) > 99:
                direction = 0
            elif center_frame_idx - interval * (N_frames - 1) < 0:
                direction = 1
            # get the neighbor list
            if direction == 1:
                neighbor_list = list(
                    range(center_frame_idx, center_frame_idx + interval * N_frames, interval))
            else:
                neighbor_list = list(
                    range(center_frame_idx, center_frame_idx - interval * N_frames, -interval))
            name_b = '{:08d}'.format(neighbor_list[0])
        else:
            # ensure not exceeding the borders
            while (center_frame_idx + self.half_N_frames * interval >
                   99) or (center_frame_idx - self.half_N_frames * interval < 0):
                center_frame_idx = random.randint(0, 99)
            # get the neighbor list
            neighbor_list = list(
                range(center_frame_idx - self.half_N_frames * interval,
                      center_frame_idx + self.half_N_frames * interval + 1, interval))
            if self.random_reverse and random.random() < 0.5:
                neighbor_list.reverse()
            name_b = '{:08d}'.format(neighbor_list[self.half_N_frames])

        assert len(
            neighbor_list) == self.opt['N_frames'], 'Wrong length of neighbor list: {}'.format(
                len(neighbor_list))
        return neighbor_list, name_b

    def read_imgs(self, root_dir, name_a, name_b, is_gt=False):
        if not isinstance(name_b, (tuple, list)):
            if self.data_type == 'lmdb':
                paths = [name_a + '_' + name_b]
            else:
                paths = [osp.join(root_dir, name_a, name_b + '.png')]
        else:
            paths = []
            for name in name_b:
                if not isinstance(name, str):
                    name = "{:08d}".format(name)
                if self.data_type == 'lmdb':
                    paths.append(name_a + '_' + name)
                else:
                    paths.append(osp.join(root_dir, name_a, name + '.png'))
        imgs = []
        for path in paths:
            if self.data_type == 'lmdb':
                if is_gt:
                    img = util.read_img(self.GT_env, path, self.GT_size_tuple)
                else:
                    img = util.read_img(self.LQ_env, path, self.LQ_size_tuple)
            else:
                img = util.read_img(None, path)
            imgs.append(img)
        return imgs

    def __getitem__(self, index):
        if self.data_type == 'lmdb' and self.GT_env is None:
            self._init_lmdb()

        LQ_size = self.opt['LQ_size']
        key = self.paths_GT[index]
        name_a, name_b = key.split('_')
        center_frame_idx = int(name_b)

        neighbor_list, name_b = self.get_neighbor_list(center_frame_idx)
        
        img_GT = self.read_imgs(self.GT_root, name_a, name_b, True)[0]
        img_LQs = self.read_imgs(self.LQ_root, name_a, neighbor_list)

        if self.opt['phase'] == 'train':
            H, W, _ = img_LQs[0].shape
            rnd_h = random.randint(0, max(0, H - LQ_size))
            rnd_w = random.randint(0, max(0, W - LQ_size))
            img_LQs = [v[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :] for v in img_LQs]
            rnd_h_HR, rnd_w_HR = int(rnd_h * self.scale), int(rnd_w * self.scale)
            GT_size = int(LQ_size * self.scale)
            img_GT = img_GT[rnd_h_HR:rnd_h_HR + GT_size, rnd_w_HR:rnd_w_HR + GT_size, :]
            # augmentation - flip, rotate
            img_LQs.append(img_GT)
            rlt = util.augment(img_LQs, self.opt['use_flip'], self.opt['use_rot'])
            img_LQs = rlt[0:-1]
            img_GT = rlt[-1]

        # stack LQ images to NHWC, N is the frame number
        img_LQs = np.stack(img_LQs, axis=0)
        # BGR => RGB
        img_GT = img_GT[:, :, [2, 1, 0]]
        img_LQs = img_LQs[:, :, :, [2, 1, 0]]
        # NHWC => NCHW
        img_GT = torch.from_numpy(np.ascontiguousarray(
                            np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQs = torch.from_numpy(np.ascontiguousarray(
                            np.transpose(img_LQs, (0, 3, 1, 2)))).float()
        return {'LQs': img_LQs, 'GT': img_GT, 'key': key, 'scale': self.scale}

    def __len__(self):
        return len(self.paths_GT)

class MultiREDSDataset(REDSDataset):
    def __init__(self, opt):
        super(MultiREDSDataset, self).__init__(opt=opt)
        self.valid_nf = opt['valid_nf']
        self.nf = opt['N_frames']

    def __getitem__(self, index):
        if self.data_type == 'lmdb' and self.GT_env is None:
            self._init_lmdb()

        LQ_size = self.opt['LQ_size']
        key = self.paths_GT[index]
        name_a, name_b = key.split('_')
        center_frame_idx = int(name_b)

        neighbor_list, name_b = self.get_neighbor_list(center_frame_idx)
        key = name_a + '_' + name_b
        
        extra = self.valid_nf // 2
        name_bs = neighbor_list[extra:self.nf-extra]

        img_GTs = self.read_imgs(self.GT_root, name_a, name_bs, True)
        img_LQs = self.read_imgs(self.LQ_root, name_a, neighbor_list)

        if self.opt['phase'] == 'train':
            H, W, _ = img_LQs[0].shape
            rnd_h = random.randint(0, max(0, H - LQ_size))
            rnd_w = random.randint(0, max(0, W - LQ_size))
            img_LQs = [v[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :] for v in img_LQs]
            rnd_h_HR, rnd_w_HR = int(rnd_h * self.scale), int(rnd_w * self.scale)
            GT_size = int(LQ_size * self.scale)
            img_GTs = [v[rnd_h_HR:rnd_h_HR + GT_size, rnd_w_HR:rnd_w_HR + GT_size, :] for v in img_GTs]
            # augmentation - flip, rotate
            num_gt = len(img_GTs)
            img_LQs += img_GTs
            img_LQs = util.augment(img_LQs, self.opt['use_flip'], self.opt['use_rot'])
            img_GTs = img_LQs[-num_gt:]
            img_LQs = img_LQs[:-num_gt]

        # stack LQ images to NHWC, N is the frame number
        img_LQs = np.stack(img_LQs, axis=0)
        img_GTs = np.stack(img_GTs, axis=0)
        # BGR => RGB
        img_GTs = img_GTs[:, :, :, [2, 1, 0]]
        img_LQs = img_LQs[:, :, :, [2, 1, 0]]
        # NHWC => NCHW
        img_GTs = torch.from_numpy(np.ascontiguousarray(
                            np.transpose(img_GTs, (0, 3, 1, 2)))).float()
        img_LQs = torch.from_numpy(np.ascontiguousarray(
                            np.transpose(img_LQs, (0, 3, 1, 2)))).float()
        return {'LQs': img_LQs, 'GT': img_GTs, 'key': key, 'scale': self.scale}

class UPREDSDataset(data.Dataset):
    def __init__(self, opt):
        super(UPREDSDataset, self).__init__()
        self.opt = opt
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        logger.info('Temporal augmentation interval list: [{}], with random reverse is {}.'.
            format(','.join(str(x) for x in opt['interval_list']), self.random_reverse))

        self.half_N_frames = opt['N_frames'] // 2
        self.GT_root = opt['dataroot_GT']
        self.data_type = self.opt['data_type']
        
        #### Load image keys
        if self.data_type == 'lmdb':
            self.paths_GT, self.GT_size_tuple = util.get_image_paths(
                                    self.data_type, self.GT_root)
            logger.info('Using lmdb meta info for cache keys.')
            self.paths_GT = [v for v in self.paths_GT if v.split('_')[0] not in \
                            ['000', '011', '015', '020']]
        else:
            seqs = sorted(os.listdir(self.GT_root))
            self.paths_GT = []
            for seq in seqs:
                if not seq in ['000', '011', '015', '020']:
                    names = os.listdir(osp.join(self.GT_root, seq))
                    self.paths_GT.extend([seq + '_' + x[:-4] for x in names])
        assert self.paths_GT, 'Error: GT path is empty.'

        if self.data_type == 'lmdb':
            self.GT_env = None

        self.scales = self.opt['scale']
        assert(len(self.scales) >= 1)
        self.GT_size = self.opt['GT_size']
        self.scales, self.LQ_sizes = self.refine_scales()
        opt['scale'] = self.scales

    def refine_scales(self):
        # def convert_scales(scales, out_size=256):
        fx = lambda x: float("{:.3f}".format(x))
        step = fx(1 / self.GT_size)
        new_scales, sample_sizes = [], []
        for scale in self.scales:
            sample_size = int(np.round(self.GT_size / scale))
            tmp_scale = fx(self.GT_size / sample_size)
            tmp_size = int(sample_size * tmp_scale)
            while tmp_size > self.GT_size:
                tmp_scale = fx(tmp_scale - step)
                tmp_size = int(sample_size * tmp_scale)
            while tmp_size < self.GT_size:
                tmp_scale = fx(tmp_scale + step)
                tmp_size = int(sample_size * tmp_scale)
            new_scales.append(tmp_scale)
            sample_sizes.append(sample_size)
        return new_scales, sample_sizes

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, 
                        readahead=False, meminit=False)

    def read_imgs(self, root_dir, name_a, name_b, scale=None):
        if scale is None:
            if self.data_type == 'lmdb':
                paths = [name_a + '_' + name_b]
            else:
                paths = [osp.join(root_dir, name_a, name_b + '.png')]
        else:
            if not isinstance(name_b, (tuple, list)):
                name_b = [name_b]
            paths = []
            for name in name_b:
                if not isinstance(name, str):
                    name = "{:08d}".format(name)
                if self.data_type == 'lmdb':
                    paths.append(name_a + '_' + name)
                else:
                    paths.append(osp.join(root_dir, name_a, name + '.png'))
        imgs = []
        for path in paths:
            if self.data_type == 'lmdb':
                if scale is None:
                    img = util.read_img(self.GT_env, path, self.GT_size_tuple)
                else:
                    img = util.read_img_to_LR(self.GT_env, path, self.GT_size_tuple, scale)
            else:
                if scale is None:
                    img = util.read_img(None, path)
                else:
                    img = util.read_img_to_LR(None, path, None, scale)
            imgs.append(img)
        return imgs

    def get_neighbor_list(self, center_frame_idx):
        #### determine the neighbor frames
        interval = random.choice(self.interval_list)
        if self.opt['border_mode']:
            direction = 1  # 1: forward; 0: backward
            N_frames = self.opt['N_frames']
            if self.random_reverse and random.random() < 0.5:
                direction = random.choice([0, 1])
            if center_frame_idx + interval * (N_frames - 1) > 99:
                direction = 0
            elif center_frame_idx - interval * (N_frames - 1) < 0:
                direction = 1
            # get the neighbor list
            if direction == 1:
                neighbor_list = list(
                    range(center_frame_idx, center_frame_idx + interval * N_frames, interval))
            else:
                neighbor_list = list(
                    range(center_frame_idx, center_frame_idx - interval * N_frames, -interval))
            name_b = '{:08d}'.format(neighbor_list[0])
        else:
            # ensure not exceeding the borders
            while (center_frame_idx + self.half_N_frames * interval >
                   99) or (center_frame_idx - self.half_N_frames * interval < 0):
                center_frame_idx = random.randint(0, 99)
            # get the neighbor list
            neighbor_list = list(
                range(center_frame_idx - self.half_N_frames * interval,
                      center_frame_idx + self.half_N_frames * interval + 1, interval))
            if self.random_reverse and random.random() < 0.5:
                neighbor_list.reverse()
            name_b = '{:08d}'.format(neighbor_list[self.half_N_frames])

        assert len(
            neighbor_list) == self.opt['N_frames'], 'Wrong length of neighbor list: {}'.format(
                len(neighbor_list))
        return neighbor_list, name_b

    def __getitem__(self, index):
        if self.data_type == 'lmdb' and self.GT_env is None:
            self._init_lmdb()

        image_index, scale_index = index
        scale = self.scales[scale_index]
        LQ_size = self.LQ_sizes[scale_index]
        key = self.paths_GT[image_index]
        name_a, name_b = key.split('_')
        center_frame_idx = int(name_b)
        key = name_a + '_' + name_b

        neighbor_list, name_b = self.get_neighbor_list(center_frame_idx)

        #### get the images
        img_GT = self.read_imgs(self.GT_root, name_a, name_b)[0]
        img_LQs = self.read_imgs(self.GT_root, name_a, neighbor_list, scale)

        if self.opt['phase'] == 'train':
            H, W, _ = img_LQs[0].shape
            rnd_h = random.randint(0, max(0, H - LQ_size))
            rnd_w = random.randint(0, max(0, W - LQ_size))
            img_LQs = [v[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :] for v in img_LQs]
            rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
            GT_size = int(LQ_size * scale)
            img_GT = img_GT[rnd_h_HR:rnd_h_HR + GT_size, rnd_w_HR:rnd_w_HR + GT_size, :]
            # augmentation - flip, rotate
            img_LQs.append(img_GT)
            rlt = util.augment(img_LQs, self.opt['use_flip'], self.opt['use_rot'])
            img_LQs = rlt[0:-1]
            img_GT = rlt[-1]

        # stack LQ images to NHWC, N is the frame number
        img_LQs = np.stack(img_LQs, axis=0)
        # BGR => RGB
        img_GT = img_GT[:, :, [2, 1, 0]]
        img_LQs = img_LQs[:, :, :, [2, 1, 0]]
        # NHWC => NCHW
        img_GT = torch.from_numpy(np.ascontiguousarray(
                            np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQs = torch.from_numpy(np.ascontiguousarray(
                            np.transpose(img_LQs, (0, 3, 1, 2)))).float()
        return {'LQs': img_LQs, 'GT': img_GT, 'key': key, 'scale': scale}

    def __len__(self):
        return len(self.paths_GT)

    def __str__(self):
        p_str = ""
        return p_str

class MetaREDSDatasetOnline(data.Dataset):
    '''
    Reading the training REDS dataset
    key example: 000_00000000
    GT: Ground-Truth;
    LQ: Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames
    support reading N LQ frames, N = 1, 3, 5, 7
    '''
    def __init__(self, opt):
        super(MetaREDSDatasetOnline, self).__init__()
        self.opt = opt
        # temporal augmentation
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        logger.info('Temporal augmentation interval list: [{}], with random reverse is {}.'.format(
            ','.join(str(x) for x in opt['interval_list']), self.random_reverse))

        self.half_N_frames = opt['N_frames'] // 2
        self.GT_root = opt['dataroot_GT']
        self.data_type = self.opt['data_type']
        
        #### Load image keys
        if self.data_type == 'lmdb':
            self.paths_GT, self.GT_size_tuple = util.get_image_paths(self.data_type, self.GT_root)
            logger.info('Using lmdb meta info for cache keys.')
            self.paths_GT = [v for v in self.paths_GT if v.split('_')[0] not in \
                            ['000', '011', '015', '020']]
        else:
            seqs = sorted(os.listdir(self.GT_root))
            self.paths_GT = []
            for seq in seqs:
                if not seq in ['000', '011', '015', '020']:
                    names = os.listdir(osp.join(self.GT_root, seq))
                    self.paths_GT.extend([seq + '_' + x[:-4] for x in names])
        assert self.paths_GT, 'Error: GT path is empty.'

        if self.data_type == 'lmdb':
            self.GT_env = None

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, 
                        readahead=False, meminit=False)

    def get_neighbor_list(self, center_frame_idx):
        #### determine the neighbor frames
        interval = random.choice(self.interval_list)
        if self.opt['border_mode']:
            direction = 1  # 1: forward; 0: backward
            N_frames = self.opt['N_frames']
            if self.random_reverse and random.random() < 0.5:
                direction = random.choice([0, 1])
            if center_frame_idx + interval * (N_frames - 1) > 99:
                direction = 0
            elif center_frame_idx - interval * (N_frames - 1) < 0:
                direction = 1
            # get the neighbor list
            if direction == 1:
                neighbor_list = list(
                    range(center_frame_idx, center_frame_idx + interval * N_frames, interval))
            else:
                neighbor_list = list(
                    range(center_frame_idx, center_frame_idx - interval * N_frames, -interval))
            name_b = '{:08d}'.format(neighbor_list[0])
        else:
            # ensure not exceeding the borders
            while (center_frame_idx + self.half_N_frames * interval >
                   99) or (center_frame_idx - self.half_N_frames * interval < 0):
                center_frame_idx = random.randint(0, 99)
            # get the neighbor list
            neighbor_list = list(
                range(center_frame_idx - self.half_N_frames * interval,
                      center_frame_idx + self.half_N_frames * interval + 1, interval))
            if self.random_reverse and random.random() < 0.5:
                neighbor_list.reverse()
            name_b = '{:08d}'.format(neighbor_list[self.half_N_frames])

        assert len(
            neighbor_list) == self.opt['N_frames'], 'Wrong length of neighbor list: {}'.format(
                len(neighbor_list))
        return neighbor_list, name_b

    def read_imgs(self, root_dir, name_a, name_b, scale=None):
        if scale is None:
            if self.data_type == 'lmdb':
                paths = [name_a + '_' + name_b]
            else:
                paths = [osp.join(root_dir, name_a, name_b + '.png')]
        else:
            if not isinstance(name_b, (tuple, list)):
                name_b = [name_b]
            paths = []
            for name in name_b:
                if not isinstance(name, str):
                    name = "{:08d}".format(name)
                if self.data_type == 'lmdb':
                    paths.append(name_a + '_' + name)
                else:
                    paths.append(osp.join(root_dir, name_a, name + '.png'))
        imgs = []
        for path in paths:
            if self.data_type == 'lmdb':
                if scale is None:
                    img = util.read_img(self.GT_env, path, self.GT_size_tuple)
                else:
                    img = util.read_img_to_LR(self.GT_env, path, self.GT_size_tuple, scale)
            else:
                if scale is None:
                    img = util.read_img(None, path)
                else:
                    img = util.read_img_to_LR(None, path, None, scale)
            imgs.append(img)
        return imgs

    def __getitem__(self, index):
        if self.data_type == 'lmdb' and self.GT_env is None:
            self._init_lmdb()

        image_index, scale_index = index
        scale = self.opt['scale'][scale_index]
        LQ_size = self.opt['LQ_size']
        key = self.paths_GT[image_index]
        name_a, name_b = key.split('_')
        center_frame_idx = int(name_b)

        neighbor_list, name_b = self.get_neighbor_list(center_frame_idx)

        #### get the images
        img_GT = self.read_imgs(self.GT_root, name_a, name_b)[0]
        img_LQs = self.read_imgs(self.GT_root, name_a, neighbor_list, scale)

        if self.opt['phase'] == 'train':
            H, W, _ = img_LQs[0].shape
            rnd_h = random.randint(0, max(0, H - LQ_size))
            rnd_w = random.randint(0, max(0, W - LQ_size))
            img_LQs = [v[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :] for v in img_LQs]
            rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
            GT_size = int(LQ_size * scale)
            img_GT = img_GT[rnd_h_HR:rnd_h_HR + GT_size, rnd_w_HR:rnd_w_HR + GT_size, :]
            # augmentation - flip, rotate
            img_LQs.append(img_GT)
            rlt = util.augment(img_LQs, self.opt['use_flip'], self.opt['use_rot'])
            img_LQs = rlt[0:-1]
            img_GT = rlt[-1]

        # stack LQ images to NHWC, N is the frame number
        img_LQs = np.stack(img_LQs, axis=0)
        # BGR => RGB
        img_GT = img_GT[:, :, [2, 1, 0]]
        img_LQs = img_LQs[:, :, :, [2, 1, 0]]
        # NHWC => NCHW
        img_GT = torch.from_numpy(np.ascontiguousarray(
                            np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQs = torch.from_numpy(np.ascontiguousarray(
                            np.transpose(img_LQs, (0, 3, 1, 2)))).float()
        return {'LQs': img_LQs, 'GT': img_GT, 'key': key, 'scale': scale}

    def __len__(self):
        return len(self.paths_GT)

class MetaREDSDataset(data.Dataset):
    '''
    Reading the training REDS dataset
    key example: 000_00000000
    GT: Ground-Truth;
    LQ: Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames
    support reading N LQ frames, N = 1, 3, 5, 7
    '''
    def __init__(self, opt):
        super(MetaREDSDataset, self).__init__()
        self.opt = opt
        # temporal augmentation
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        logger.info('Temporal augmentation interval list: [{}], with random reverse is {}.'.format(
            ','.join(str(x) for x in opt['interval_list']), self.random_reverse))

        self.half_N_frames = opt['N_frames'] // 2
        self.GT_root = opt['dataroot_GT']
        self.LQ_root = opt['dataroot_LQ']
        self.scales = opt['scale']
        assert(len(self.scales) >= 1)

        seqs = sorted(os.listdir(self.GT_root))
        self.paths_GT = []
        for seq in seqs:
            if not seq in ['000', '011', '015', '020']:
                names = os.listdir(osp.join(self.GT_root, seq))
                self.paths_GT.extend([seq + '_' + x[:-4] for x in names])

        assert self.paths_GT, 'Error: GT path is empty.'

    def get_neighbor_list(self, center_frame_idx):
        #### determine the neighbor frames
        interval = random.choice(self.interval_list)
        if self.opt['border_mode']:
            direction = 1  # 1: forward; 0: backward
            N_frames = self.opt['N_frames']
            if self.random_reverse and random.random() < 0.5:
                direction = random.choice([0, 1])
            if center_frame_idx + interval * (N_frames - 1) > 99:
                direction = 0
            elif center_frame_idx - interval * (N_frames - 1) < 0:
                direction = 1
            # get the neighbor list
            if direction == 1:
                neighbor_list = list(
                    range(center_frame_idx, center_frame_idx + interval * N_frames, interval))
            else:
                neighbor_list = list(
                    range(center_frame_idx, center_frame_idx - interval * N_frames, -interval))
            name_b = '{:08d}'.format(neighbor_list[0])
        else:
            # ensure not exceeding the borders
            while (center_frame_idx + self.half_N_frames * interval >
                   99) or (center_frame_idx - self.half_N_frames * interval < 0):
                center_frame_idx = random.randint(0, 99)
            # get the neighbor list
            neighbor_list = list(
                range(center_frame_idx - self.half_N_frames * interval,
                      center_frame_idx + self.half_N_frames * interval + 1, interval))
            if self.random_reverse and random.random() < 0.5:
                neighbor_list.reverse()
            name_b = '{:08d}'.format(neighbor_list[self.half_N_frames])

        assert len(
            neighbor_list) == self.opt['N_frames'], 'Wrong length of neighbor list: {}'.format(
                len(neighbor_list))
        return neighbor_list, name_b

    def read_imgs(self, root_dir, name_a, name_b, scale=None):
        if scale is None:
            paths = [osp.join(root_dir, name_a, name_b + '.png')]
        else:
            if not isinstance(name_b, (tuple, list)):
                name_b = [name_b]
            paths = []
            for name in name_b:
                if not isinstance(name, str):
                    name = "{:08d}".format(name)
                paths.append(osp.join(root_dir, 'X{:.02f}'.format(scale), 
                    name_a, name+'.png'))
        imgs = []
        for path in paths:
            imgs.append(util.read_img(None, path))
        return imgs

    def __getitem__(self, index):
        image_index, scale_index = index
        scale = self.scales[scale_index]
        LQ_size = self.opt['LQ_size']
        key = self.paths_GT[image_index]
        name_a, name_b = key.split('_')
        center_frame_idx = int(name_b)

        neighbor_list, name_b = self.get_neighbor_list(center_frame_idx)

        #### get the images
        img_GT = self.read_imgs(self.GT_root, name_a, name_b)[0]
        img_LQs = self.read_imgs(self.LQ_root, name_a, neighbor_list, scale)

        if self.opt['phase'] == 'train':
            H, W, _ = img_LQs[0].shape
            rnd_h = random.randint(0, max(0, H - LQ_size))
            rnd_w = random.randint(0, max(0, W - LQ_size))
            img_LQs = [v[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :] for v in img_LQs]
            rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
            GT_size = int(LQ_size * scale)
            img_GT = img_GT[rnd_h_HR:rnd_h_HR + GT_size, rnd_w_HR:rnd_w_HR + GT_size, :]
            # augmentation - flip, rotate
            img_LQs.append(img_GT)
            rlt = util.augment(img_LQs, self.opt['use_flip'], self.opt['use_rot'])
            img_LQs = rlt[0:-1]
            img_GT = rlt[-1]

        # stack LQ images to NHWC, N is the frame number
        img_LQs = np.stack(img_LQs, axis=0)
        # BGR => RGB
        img_GT = img_GT[:, :, [2, 1, 0]]
        img_LQs = img_LQs[:, :, :, [2, 1, 0]]
        # NHWC => NCHW
        img_GT = torch.from_numpy(np.ascontiguousarray(
                            np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQs = torch.from_numpy(np.ascontiguousarray(
                            np.transpose(img_LQs, (0, 3, 1, 2)))).float()
        return {'LQs': img_LQs, 'GT': img_GT, 'key': key, 'scale': scale}

    def __len__(self):
        return len(self.paths_GT)
