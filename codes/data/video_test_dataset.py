import os
import os.path as osp
import pickle
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import data.util as util

class VideoTestDataset(data.Dataset):
    """
    A video test dataset. Support:
    Vid4
    REDS4
    Vimeo90K-Test

    no need to prepare LMDB files
    """

    def __init__(self, opt):
        super(VideoTestDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.half_N_frames = opt['N_frames'] // 2
        self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.data_type = self.opt['data_type']
        self.data_info = {'path_LQ': [], 'path_GT': [], 'folder': [], 'idx': [], 'border': []}
        if self.data_type == 'lmdb':
            raise ValueError('No need to use LMDB during validation/test.')
        #### Generate data info and cache data
        self.scale = opt['scale'][-1] if isinstance(opt['scale'], (tuple, list)) else opt['scale']
        self.imgs_LQ, self.imgs_GT = {}, {}
        if opt['name'].lower() in ['vid4', 'reds4']:
            subfolders_LQ = util.glob_file_list(self.LQ_root)
            subfolders_GT = util.glob_file_list(self.GT_root)
            for subfolder_LQ, subfolder_GT in zip(subfolders_LQ, subfolders_GT):
                subfolder_name = osp.basename(subfolder_GT)
                img_paths_LQ = util.glob_file_list(subfolder_LQ)
                img_paths_GT = util.glob_file_list(subfolder_GT)
                max_idx = len(img_paths_LQ)
                assert max_idx == len(
                    img_paths_GT), 'Different number of images in LQ and GT folders'
                self.data_info['path_LQ'].extend(img_paths_LQ)
                self.data_info['path_GT'].extend(img_paths_GT)
                self.data_info['folder'].extend([subfolder_name] * max_idx)
                for i in range(max_idx):
                    self.data_info['idx'].append('{}/{}'.format(i, max_idx))
                border_l = [0] * max_idx
                for i in range(self.half_N_frames):
                    border_l[i] = 1
                    border_l[max_idx - i - 1] = 1
                self.data_info['border'].extend(border_l)

                if self.cache_data:
                    self.imgs_LQ[subfolder_name] = util.read_img_seq(img_paths_LQ)
                    self.imgs_GT[subfolder_name] = util.read_img_seq(img_paths_GT)
        elif opt['name'].lower() in ['vimeo90k-test']:
            pass  # TODO
        else:
            raise ValueError(
                'Not support video test dataset. Support Vid4, REDS4 and Vimeo90k-Test.')

    def __getitem__(self, index):
        # path_LQ = self.data_info['path_LQ'][index]
        # path_GT = self.data_info['path_GT'][index]
        scale = self.scale
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]

        if self.cache_data:
            select_idx = util.index_generation(idx, max_idx, self.opt['N_frames'],
                                               padding=self.opt['padding'])
            imgs_LQ = self.imgs_LQ[folder].index_select(0, torch.LongTensor(select_idx))
            img_GT = self.imgs_GT[folder][idx]
            if idx > 0:
                img_preGT = self.imgs_GT[folder][idx - 1]
        else:
            pass  # TODO

        out = {'LQs': imgs_LQ, 'GT': img_GT, 'folder': folder, 'scale': scale, 
              'idx': self.data_info['idx'][index], 'border': border, }
        if idx > 0:
            out['Pre'] = img_preGT
        return out

    def __len__(self):
        return len(self.data_info['path_GT'])

class OnlineVideoTestDataset(data.Dataset):
    def __init__(self, opt):
        super(OnlineVideoTestDataset, self).__init__()
        self.opt = opt
        self.data_name = opt['name']
        self.cache_data = opt['cache_data']
        self.half_N_frames = opt['N_frames'] // 2
        self.GT_root = opt['dataroot_GT']
        self.is_lmdb = self.GT_root.endswith('lmdb')

        self.data_info = self.data_info = {'path_GT': [], 'idx': [], 
                                           'folder': [], 'border': []}

        self.scale = opt['scale']
        self.imgs_GT = {}
        if self.data_name.lower() in ['vid4', 'reds4']:
            if self.is_lmdb:
                raise TypeError("{} data should not lmdb".format(self.data_name))
            subs = sorted(os.listdir(self.GT_root))
            for sub in subs:
                sub_dir = osp.join(self.GT_root, sub)
                im_names = sorted(os.listdir(sub_dir))
                im_paths = [osp.join(sub_dir, name) for name in im_names]
                max_idx = len(im_names)
                self.data_info['path_GT'].extend(im_paths)
                self.data_info['folder'].extend([sub] * max_idx)
                for i in range(max_idx):
                    self.data_info['idx'].append('{}/{}'.format(i, max_idx))
                border_l = [0] * max_idx
                for i in range(self.half_N_frames):
                    border_l[i] = 1
                    border_l[max_idx - i - 1] = 1
                self.data_info['border'].extend(border_l)
                if self.cache_data:
                    self.imgs_GT[sub] = self.read_sub_images(im_paths)

        elif self.data_name.lower() in ['vimeo', 'vimeo90k', 'vimeo90k-test']:
            if self.is_lmdb:
                paths_GT, self.GT_size_tuple = util.get_image_paths('lmdb', self.GT_root)
                self.data_info['path_GT'] = [x for x in paths_GT if x.endswith('_4')]
            else:
                split_file = osp.join(self.GT_root, 'sep_trainlist.txt')
                paths_GT = []
                with open(txt_file, 'r') as f:
                    lines = f.readlines()
                    img_list = [line.strip() for line in lines]
                for item in img_list:
                    key = osp.join(*item.split('/'), 'im4.png')
                    paths_GT.append(osp.join(self.GT_root, 'sequences', key))
                self.data_info['path_GT'] = paths_GT
        else:
            raise ValueError(
                'Not support video test dataset. Support Vid4, REDS4 and Vimeo90k-Test.')

    def read_sub_images(self, paths):
        if not isinstance(paths, (list, tuple)):
            paths = [paths]
        imgs = []
        for path in paths:
            imgs.append(util.read_img(None, path, dtype='uint8'))
        return np.stack(imgs)

    def get_effect_H_W(self, H, W, scale):
        pass

    def __getitem__(self, index):
        scale = self.scale
        if self.data_name.lower() in ['vid4', 'reds4']:
            folder = self.data_info['folder'][index]
            idx, max_idx = self.data_info['idx'][index].split('/')
            idx, max_idx = int(idx), int(max_idx)
            border = self.data_info['border'][index]
            
            select_idx = util.index_generation(idx, max_idx, self.opt['N_frames'],
                                               padding=self.opt['padding'])
            if self.cache_data:
                imgs = self.imgs_GT[folder][select_idx]
                N, H, W, C = imgs.shape

                img_GT = imgs[self.half_N_frames] / 255.
                imgs_LQ = [np.array(Image.fromarray(img).resize((LQ_W, LQ_H), 
                        Image.BICUBIC)) / 255. for img in imgs]
            else:
                pass  # TODO

            out = {'LQs': imgs_LQ, 'GT': img_GT, 'folder': folder, 'scale': scale, 
                  'idx': self.data_info['idx'][index], 'border': border, }
        else:
            key = self.data_info['path_GT'][index]


    def __len__(self):
        return len(self.data_info['path_GT'])


class ImgTestDataset(data.Dataset):
    """
    A video test dataset. Support:
    Vid4
    REDS4
    Vimeo90K-Test

    no need to prepare LMDB files
    """

    def __init__(self, opt):
        super(ImgTestDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.half_N_frames = opt['N_frames'] // 2
        self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.data_info = {'path_LQ': [], 'path_GT': [], 'folder': [], 'idx': [], 'border': []}

        #### Generate data info and cache data
        self.scale = opt['scale'][-1] if isinstance(opt['scale'], (tuple, list)) else opt['scale']
        self.imgs_LQ, self.imgs_GT = {}, {}
        if opt['name'].lower() in ['vid4', 'reds4']:
            subfolders_LQ = util.glob_file_list(self.LQ_root)
            subfolders_GT = util.glob_file_list(self.GT_root)
            for subfolder_LQ, subfolder_GT in zip(subfolders_LQ, subfolders_GT):
                subfolder_name = osp.basename(subfolder_GT)
                img_paths_LQ = util.glob_file_list(subfolder_LQ)
                img_paths_GT = util.glob_file_list(subfolder_GT)
                max_idx = len(img_paths_LQ)
                assert max_idx == len(
                    img_paths_GT), 'Different number of images in LQ and GT folders'
                self.data_info['path_LQ'].extend(img_paths_LQ)
                self.data_info['path_GT'].extend(img_paths_GT)
                self.data_info['folder'].extend([subfolder_name] * max_idx)
                for i in range(max_idx):
                    self.data_info['idx'].append('{}/{}'.format(i, max_idx))
                border_l = [0] * max_idx
                for i in range(self.half_N_frames):
                    border_l[i] = 1
                    border_l[max_idx - i - 1] = 1
                self.data_info['border'].extend(border_l)

                if self.cache_data:
                    self.imgs_LQ[subfolder_name] = util.read_img_seq(img_paths_LQ)
                    self.imgs_GT[subfolder_name] = util.read_img_seq(img_paths_GT)
        elif opt['name'].lower() in ['vimeo90k-test']:
            pass  # TODO
        else:
            raise ValueError(
                'Not support video test dataset. Support Vid4, REDS4 and Vimeo90k-Test.')

    def __getitem__(self, index):
        # path_LQ = self.data_info['path_LQ'][index]
        # path_GT = self.data_info['path_GT'][index]
        scale = self.scale
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]

        if self.cache_data:
            # select_idx = util.index_generation(idx, max_idx, self.opt['N_frames'],
            #                                    padding=self.opt['padding'])
            # imgs_LQ = self.imgs_LQ[folder].index_select(0, torch.LongTensor(select_idx))
            imgs_LQ = self.imgs_LQ[folder][idx]
            img_GT = self.imgs_GT[folder][idx]
        else:
            pass  # TODO

        return {
            'LQs': imgs_LQ,
            'GT': img_GT,
            'folder': folder,
            'idx': self.data_info['idx'][index],
            'border': border, 
            'scale': scale
        }

    def __len__(self):
        return len(self.data_info['path_GT'])

class MultiImgTestDataset(data.Dataset):
    """
    A video test dataset. Support:
    Vid4
    REDS4
    Vimeo90K-Test

    no need to prepare LMDB files
    """

    def __init__(self, opt):
        super(MultiImgTestDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.data_info = {'path_LQ': [], 'path_GT': [], 'folder': [], 'idx': []}

        #### Generate data info and cache data
        self.scale = opt['scale'][-1] if isinstance(opt['scale'], (tuple, list)) else opt['scale']
        self.imgs_LQ, self.imgs_GT = {}, {}
        if opt['name'].lower() in ['vid4', 'reds4']:
            subfolders_LQ = util.glob_file_list(self.LQ_root)
            subfolders_GT = util.glob_file_list(self.GT_root)
            for subfolder_LQ, subfolder_GT in zip(subfolders_LQ, subfolders_GT):
                subfolder_name = osp.basename(subfolder_GT)
                img_paths_LQ = util.glob_file_list(subfolder_LQ)
                img_paths_GT = util.glob_file_list(subfolder_GT)
                max_idx = len(img_paths_LQ)
                assert max_idx == len(
                    img_paths_GT), 'Different number of images in LQ and GT folders'
                self.data_info['path_LQ'].extend(img_paths_LQ)
                self.data_info['path_GT'].extend(img_paths_GT)
                self.data_info['folder'].extend([subfolder_name] * max_idx)
                for i in range(max_idx):
                    self.data_info['idx'].append('{}/{}'.format(i, max_idx))

                if self.cache_data:
                    self.imgs_LQ[subfolder_name] = util.read_img_seq(img_paths_LQ)
                    self.imgs_GT[subfolder_name] = util.read_img_seq(img_paths_GT)
        elif opt['name'].lower() in ['vimeo90k-test']:
            pass  # TODO
        else:
            raise ValueError(
                'Not support video test dataset. Support Vid4, REDS4 and Vimeo90k-Test.')

    def __getitem__(self, index):
        # path_LQ = self.data_info['path_LQ'][index]
        # path_GT = self.data_info['path_GT'][index]
        scale = self.scale
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)

        if self.cache_data:
            imgs_LQ = self.imgs_LQ[folder][idx]
            img_GT = self.imgs_GT[folder][idx]
        else:
            pass  # TODO

        return {
            'LQs': imgs_LQ,
            'GT': img_GT,
            'folder': folder,
            'idx': self.data_info['idx'][index],
            'scale': scale
        }

    def __len__(self):
        return len(self.data_info['path_GT'])
