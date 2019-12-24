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
from PIL import Image
import torch.utils.data as data
import data.util as util
try:
    import mc  # import memcached
except ImportError:
    pass

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
        self.LR_input = False if opt['GT_size'] == opt['LQ_size'] else True  # low resolution inputs
        #### directly load image keys
        if self.data_type == 'lmdb':
            self.paths_GT, _ = util.get_image_paths(self.data_type, opt['dataroot_GT'])
            logger.info('Using lmdb meta info for cache keys.')
        elif opt['cache_keys']:
            logger.info('Using cache keys: {}'.format(opt['cache_keys']))
            self.paths_GT = pickle.load(open(opt['cache_keys'], 'rb'))['keys']
        else:
            raise ValueError(
                'Need to create cache keys (meta_info.pkl) by running [create_lmdb.py]')

        # remove the REDS4 for testing
        self.paths_GT = [
            v for v in self.paths_GT if v.split('_')[0] not in ['000', '011', '015', '020']
        ]
        assert self.paths_GT, 'Error: GT path is empty.'

        if self.data_type == 'lmdb':
            self.GT_env, self.LQ_env = None, None
        elif self.data_type == 'mc':  # memcached
            self.mclient = None
        elif self.data_type == 'img':
            pass
        else:
            raise ValueError('Wrong data type: {}'.format(self.data_type))

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
                                meminit=False)

    def _ensure_memcached(self):
        if self.mclient is None:
            # specify the config files
            server_list_config_file = None
            client_config_file = None
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file,
                                                          client_config_file)

    def _read_img_mc(self, path):
        ''' Return BGR, HWC, [0, 255], uint8'''
        value = mc.pyvector()
        self.mclient.Get(path, value)
        value_buf = mc.ConvertBuffer(value)
        img_array = np.frombuffer(value_buf, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        return img

    def _read_img_mc_BGR(self, path, name_a, name_b):
        ''' Read BGR channels separately and then combine for 1M limits in cluster'''
        img_B = self._read_img_mc(osp.join(path + '_B', name_a, name_b + '.png'))
        img_G = self._read_img_mc(osp.join(path + '_G', name_a, name_b + '.png'))
        img_R = self._read_img_mc(osp.join(path + '_R', name_a, name_b + '.png'))
        img = cv2.merge((img_B, img_G, img_R))
        return img

    def __getitem__(self, index):
        if self.data_type == 'mc':
            self._ensure_memcached()
        elif self.data_type == 'lmdb' and (self.GT_env is None or self.LQ_env is None):
            self._init_lmdb()

        scale = self.opt['scale']
        GT_size = self.opt['GT_size']
        key = self.paths_GT[index]
        name_a, name_b = key.split('_')
        center_frame_idx = int(name_b)

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

        #### get the GT image (as the center frame)
        if self.data_type == 'mc':
            img_GT = self._read_img_mc_BGR(self.GT_root, name_a, name_b)
            img_GT = img_GT.astype(np.float32) / 255.
        elif self.data_type == 'lmdb':
            img_GT = util.read_img(self.GT_env, key, (3, 720, 1280))
        else:
            img_GT = util.read_img(None, osp.join(self.GT_root, name_a, name_b + '.png'))

        #### get LQ images
        LQ_size_tuple = (3, 180, 320) if self.LR_input else (3, 720, 1280)
        img_LQ_l = []
        for v in neighbor_list:
            img_LQ_path = osp.join(self.LQ_root, name_a, '{:08d}.png'.format(v))
            if self.data_type == 'mc':
                if self.LR_input:
                    img_LQ = self._read_img_mc(img_LQ_path)
                else:
                    img_LQ = self._read_img_mc_BGR(self.LQ_root, name_a, '{:08d}'.format(v))
                img_LQ = img_LQ.astype(np.float32) / 255.
            elif self.data_type == 'lmdb':
                img_LQ = util.read_img(self.LQ_env, '{}_{:08d}'.format(name_a, v), LQ_size_tuple)
            else:
                img_LQ = util.read_img(None, img_LQ_path)
            img_LQ_l.append(img_LQ)

        if self.opt['phase'] == 'train':
            C, H, W = LQ_size_tuple  # LQ size
            # randomly crop
            if self.LR_input:
                LQ_size = GT_size // scale
                rnd_h = random.randint(0, max(0, H - LQ_size))
                rnd_w = random.randint(0, max(0, W - LQ_size))
                img_LQ_l = [v[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :] for v in img_LQ_l]
                rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
                img_GT = img_GT[rnd_h_HR:rnd_h_HR + GT_size, rnd_w_HR:rnd_w_HR + GT_size, :]
            else:
                rnd_h = random.randint(0, max(0, H - GT_size))
                rnd_w = random.randint(0, max(0, W - GT_size))
                img_LQ_l = [v[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :] for v in img_LQ_l]
                img_GT = img_GT[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :]

            # augmentation - flip, rotate
            img_LQ_l.append(img_GT)
            rlt = util.augment(img_LQ_l, self.opt['use_flip'], self.opt['use_rot'])
            img_LQ_l = rlt[0:-1]
            img_GT = rlt[-1]

        # stack LQ images to NHWC, N is the frame number
        img_LQs = np.stack(img_LQ_l, axis=0)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_GT = img_GT[:, :, [2, 1, 0]]
        img_LQs = img_LQs[:, :, :, [2, 1, 0]]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs,
                                                                     (0, 3, 1, 2)))).float()
        return {'LQs': img_LQs, 'GT': img_GT, 'key': key}

    def __len__(self):
        return len(self.paths_GT)

class MetaREDSDatasetV0(data.Dataset):
    '''
    Reading the training REDS dataset
    key example: 000_00000000
    GT: Ground-Truth;
    LQ: Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames
    support reading N LQ frames, N = 1, 3, 5, 7
    '''

    def __init__(self, opt):
        super(MetaREDSDatasetV0, self).__init__()
        self.opt = opt
        # temporal augmentation
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        logger.info('Temporal augmentation interval list: [{}], with random reverse is {}.'.format(
            ','.join(str(x) for x in opt['interval_list']), self.random_reverse))

        self.half_N_frames = opt['N_frames'] // 2
        self.GT_root = opt['dataroot_GT']
        self.data_type = self.opt['data_type']
        
        #### directly load image keys
        if self.data_type == 'lmdb':
            self.paths_GT, _ = util.get_image_paths(self.data_type, opt['dataroot_GT'])
            logger.info('Using lmdb meta info for cache keys.')
        elif opt['cache_keys']:
            logger.info('Using cache keys: {}'.format(opt['cache_keys']))
            self.paths_GT = pickle.load(open(opt['cache_keys'], 'rb'))['keys']
        else:
            raise ValueError(
                'Need to create cache keys (meta_info.pkl) by running [create_lmdb.py]')

        # remove the REDS4 for testing
        self.paths_GT = [
            v for v in self.paths_GT if v.split('_')[0] not in ['000', '011', '015', '020']
        ]
        assert self.paths_GT, 'Error: GT path is empty.'

        if self.data_type == 'lmdb':
            self.GT_env = None
        elif self.data_type == 'mc':  # memcached
            self.mclient = None
        elif self.data_type == 'img':
            pass
        else:
            raise ValueError('Wrong data type: {}'.format(self.data_type))

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                meminit=False)

    def _ensure_memcached(self):
        if self.mclient is None:
            # specify the config files
            server_list_config_file = None
            client_config_file = None
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file,
                                                          client_config_file)

    def _read_img_mc(self, path):
        ''' Return BGR, HWC, [0, 255], uint8'''
        value = mc.pyvector()
        self.mclient.Get(path, value)
        value_buf = mc.ConvertBuffer(value)
        img_array = np.frombuffer(value_buf, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        return img

    def _read_img_mc_BGR(self, path, name_a, name_b):
        ''' Read BGR channels separately and then combine for 1M limits in cluster'''
        img_B = self._read_img_mc(osp.join(path + '_B', name_a, name_b + '.png'))
        img_G = self._read_img_mc(osp.join(path + '_G', name_a, name_b + '.png'))
        img_R = self._read_img_mc(osp.join(path + '_R', name_a, name_b + '.png'))
        img = cv2.merge((img_B, img_G, img_R))
        return img

    def __getitem__(self, index):
        if self.data_type == 'mc':
            self._ensure_memcached()
        elif self.data_type == 'lmdb' and self.GT_env is None:
            self._init_lmdb()

        image_index, scale_index = index
        scale = self.opt['scale'][scale_index]
        GT_size = self.opt['GT_size']
        LQ_size = self.opt['LQ_size']
        key = self.paths_GT[image_index]
        name_a, name_b = key.split('_')
        center_frame_idx = int(name_b)

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

        #### get the GT image (as the center frame)
        if self.data_type == 'mc':
            img_GT = self._read_img_mc_BGR(self.GT_root, name_a, name_b)
            img_GT = img_GT.astype(np.float32) / 255.
        elif self.data_type == 'lmdb':
            img_GT = util.read_img(self.GT_env, key, (3, 720, 1280))
        else:
            img_GT = util.read_img(None, osp.join(self.GT_root, name_a, name_b + '.png'))

        #### get LQ images
        img_LQ_l = []
        for v in neighbor_list:
            img_LQ_path = osp.join(self.GT_root, name_a, '{:08d}.png'.format(v))
            if self.data_type == 'mc':
                img_LQ = self._read_img_mc_BGR(self.GT_root, name_a, '{:08d}'.format(v))
                img_LQ = img_LQ.astype(np.float32) / 255.
            elif self.data_type == 'lmdb':
                img_LQ = util.read_img_to_LR(self.GT_env, '{}_{:08d}'.format(name_a, v), 
                                (3, 720, 1280), scale)
            else:
                img_LQ = util.read_img_to_LR(None, img_LQ_path, None, scale)
            img_LQ_l.append(img_LQ)

        if self.opt['phase'] == 'train':
            img_GT, img_LQ_l = util.get_img_patch(img_GT, img_LQ_l, LQ_size, scale)
            # augmentation - flip, rotate
            img_LQ_l.append(img_GT)
            rlt = util.augment(img_LQ_l, self.opt['use_flip'], self.opt['use_rot'])
            img_LQ_l = rlt[0:-1]
            img_GT = rlt[-1]

        # stack LQ images to NHWC, N is the frame number
        img_LQs = np.stack(img_LQ_l, axis=0)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_GT = img_GT[:, :, [2, 1, 0]]
        img_LQs = img_LQs[:, :, :, [2, 1, 0]]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs,
                                                                     (0, 3, 1, 2)))).float()
        return {'LQs': img_LQs, 'GT': img_GT, 'key': key, 'scale': scale}

    def __len__(self):
        return len(self.paths_GT)

class MetaREDSDatasetV1(data.Dataset):
    '''
    Reading the training REDS dataset
    key example: 000_00000000
    GT: Ground-Truth;
    LQ: Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames
    support reading N LQ frames, N = 1, 3, 5, 7
    '''

    def __init__(self, opt):
        super(MetaREDSDatasetV1, self).__init__()
        self.opt = opt
        # temporal augmentation
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        logger.info('Temporal augmentation interval list: [{}], with random reverse is {}.'.format(
            ','.join(str(x) for x in opt['interval_list']), self.random_reverse))

        self.half_N_frames = opt['N_frames'] // 2
        self.GT_root = opt['dataroot_GT']
        self.LQ_root = opt['dataroot_LQ']
        self.data_type = self.opt['data_type']
        self.scales = opt['scale']
        assert(len(self.scales) > 1)

        #### directly load image keys
        if self.data_type == 'lmdb':
            self.paths_GT, self.GT_size_tuple = util.get_image_paths(self.data_type, self.GT_root)
            with open(osp.join(self.LQ_root, 'meta_info.pkl'), 'rb') as f:
                meta_info = pickle.load(f)
                sizes = meta_info['resolution']
            self.LQ_size_tuple = {k:(v[2],v[0],v[1]) for k,v in sizes.items()}
            logger.info('Using lmdb meta info for cache keys.')
        elif opt['cache_keys']:
            logger.info('Using cache keys: {}'.format(opt['cache_keys']))
            self.paths_GT = pickle.load(open(opt['cache_keys'], 'rb'))['keys']
        else:
            raise ValueError(
                'Need to create cache keys (meta_info.pkl) by running [create_lmdb.py]')

        # remove the REDS4 for testing
        self.paths_GT = [
            v for v in self.paths_GT if v.split('_')[0] not in ['000', '011', '015', '020']
        ]
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
        return neighbor_list

    def __getitem__(self, index):
        if self.data_type == 'lmdb' and self.GT_env is None:
            self._init_lmdb()

        image_index, scale_index = index
        scale = self.scales[scale_index]
        LQ_size = self.opt['LQ_size']
        key = self.paths_GT[image_index]
        name_a, name_b = key.split('_')
        center_frame_idx = int(name_b)

        neighbor_list = self.get_neighbor_list(center_frame_idx)

        #### get the GT image (as the center frame)
        if self.data_type == 'lmdb':
            img_GT = util.read_img(self.GT_env, key, self.GT_size_tuple)
        else:
            img_GT = util.read_img(None, osp.join(self.GT_root, name_a, name_b + '.png'))

        #### get LQ images
        img_LQ_l = []
        for v in neighbor_list:
            img_LQ_path = osp.join(self.LQ_root, 'X{:.02f}'.format(scale), 
                                   name_a, '{:08d}.png'.format(v))
            if self.data_type == 'lmdb':
                LQ_prefix = 'X{:.02f}'.format(scale)
                img_LQ = util.read_img(self.LQ_env, '{}_{}_{:08d}'.format(LQ_prefix, name_a, v), 
                                       self.LQ_size_tuple[LQ_prefix])
            else:
                # img_LQ = util.read_img_to_LR(None, img_LQ_path, None, scale)
                img_LQ = util.read_img(None, img_LQ_path)
            img_LQ_l.append(img_LQ)

        if self.opt['phase'] == 'train':
            img_GT, img_LQ_l = util.get_img_patch(img_GT, img_LQ_l, LQ_size, scale)
            # augmentation - flip, rotate
            img_LQ_l.append(img_GT)
            rlt = util.augment(img_LQ_l, self.opt['use_flip'], self.opt['use_rot'])
            img_LQ_l = rlt[0:-1]
            img_GT = rlt[-1]

        # stack LQ images to NHWC, N is the frame number
        img_LQs = np.stack(img_LQ_l, axis=0)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_GT = img_GT[:, :, [2, 1, 0]]
        img_LQs = img_LQs[:, :, :, [2, 1, 0]]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs,
                                                                     (0, 3, 1, 2)))).float()
        # import matplotlib.pyplot as plt
        # plt.imshow(img_GT.numpy().transpose((1,2,0)))
        # plt.figure(2)
        # plt.imshow(img_LQs[0].numpy().transpose((1,2,0))) 
        # plt.show()
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
        assert(len(self.scales) > 1)

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
        return neighbor_list

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
            imgs.append(np.array(Image.open(path))/255.)
        return imgs

    def __getitem__(self, index):
        image_index, scale_index = index
        scale = self.scales[scale_index]
        LQ_size = self.opt['LQ_size']
        key = self.paths_GT[image_index]
        name_a, name_b = key.split('_')
        center_frame_idx = int(name_b)

        neighbor_list = self.get_neighbor_list(center_frame_idx)

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
        # HWC => CHW
        img_GT = torch.from_numpy(np.ascontiguousarray(
                            np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQs = torch.from_numpy(np.ascontiguousarray(
                            np.transpose(img_LQs, (0, 3, 1, 2)))).float()
        return {'LQs': img_LQs, 'GT': img_GT, 'key': key, 'scale': scale}

    def __len__(self):
        return len(self.paths_GT)
