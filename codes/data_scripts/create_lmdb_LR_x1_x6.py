"""Create lmdb files for [General images (291 images/DIV2K) | Vimeo90K | REDS] training datasets"""

import os, sys
import os.path as osp
import glob
import pickle
from multiprocessing import Pool
import numpy as np
import lmdb
import cv2
import argparse

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import data.util as data_util  # noqa: E402
import utils.util as util  # noqa: E402

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

def main_v0(cfgs):
    if not osp.exists(cfgs.dst_dir):
        os.makedirs(cfgs.dst_dir)
    if cfgs.dataset == 'vimeo90k':
        vimeo90k(cfgs)
    elif cfgs.dataset == 'REDS':
        scale = cfgs.scale
        img_folder = osp.join(cfgs.src_dir, "X{:.02f}".format(scale))
        lmdb_save_path = osp.join(cfgs.dst_dir, "X{:.02f}.lmdb".format(scale))
        one_example = osp.join(img_folder, '000', '00000000.png')
        im_example = cv2.imread(one_example)
        H, W, _ = im_example.shape
        REDS(img_folder, lmdb_save_path, H, W, scale)
    elif cfgs.dataset == 'general':
        opt = {}
        opt['img_folder'] = '../../datasets/DIV2K/DIV2K800_sub'
        opt['lmdb_save_path'] = '../../datasets/DIV2K/DIV2K800_sub.lmdb'
        opt['name'] = 'DIV2K800_sub_GT'
        general_image_folder(opt)
    elif cfgs.dataset == 'DIV2K_demo':
        opt = {}
        ## GT
        opt['img_folder'] = '../../datasets/DIV2K/DIV2K800_sub'
        opt['lmdb_save_path'] = '../../datasets/DIV2K/DIV2K800_sub.lmdb'
        opt['name'] = 'DIV2K800_sub_GT'
        general_image_folder(opt)
        ## LR
        opt['img_folder'] = '../../datasets/DIV2K/DIV2K800_sub_bicLRx4'
        opt['lmdb_save_path'] = '../../datasets/DIV2K/DIV2K800_sub_bicLRx4.lmdb'
        opt['name'] = 'DIV2K800_sub_bicLRx4'
        general_image_folder(opt)
    elif cfgs.dataset == 'test':
        dataroot = osp.join(cfgs.dst_dir, "X{:.02f}.lmdb".format(cfgs.scale))
        test_lmdb(dataroot=dataroot, dataset='REDS')

def main_old():
    dataset = 'REDS'  # vimeo90K | REDS | general (e.g., DIV2K, 291) | DIV2K_demo |test
    mode = 'train_sharp'  # used for vimeo90k and REDS datasets
    # vimeo90k: GT | LR | flow
    # REDS: train_sharp, train_sharp_bicubic, train_blur_bicubic, train_blur, train_blur_comp
    #       train_sharp_flowx4
    if dataset == 'vimeo90k':
        vimeo90k(mode)
    elif dataset == 'REDS':
        REDS(mode)
    elif dataset == 'general':
        opt = {}
        opt['img_folder'] = '../../datasets/DIV2K/DIV2K800_sub'
        opt['lmdb_save_path'] = '../../datasets/DIV2K/DIV2K800_sub.lmdb'
        opt['name'] = 'DIV2K800_sub_GT'
        general_image_folder(opt)
    elif dataset == 'DIV2K_demo':
        opt = {}
        ## GT
        opt['img_folder'] = '../../datasets/DIV2K/DIV2K800_sub'
        opt['lmdb_save_path'] = '../../datasets/DIV2K/DIV2K800_sub.lmdb'
        opt['name'] = 'DIV2K800_sub_GT'
        general_image_folder(opt)
        ## LR
        opt['img_folder'] = '../../datasets/DIV2K/DIV2K800_sub_bicLRx4'
        opt['lmdb_save_path'] = '../../datasets/DIV2K/DIV2K800_sub_bicLRx4.lmdb'
        opt['name'] = 'DIV2K800_sub_bicLRx4'
        general_image_folder(opt)
    elif dataset == 'test':
        test_lmdb('../../datasets/REDS/train_sharp_wval.lmdb', 'REDS')

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_paths_from_images(path):
    """get image path list from image folder"""
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images

def read_image_worker(path, key):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return (key, img)

def general_image_folder(opt):
    """Create lmdb for general image folders
    Users should define the keys, such as: '0321_s035' for DIV2K sub-images
    If all the images have the same resolution, it will only store one copy of resolution info.
        Otherwise, it will store every resolution info.
    """
    #### configurations
    read_all_imgs = False  # whether real all images to memory with multiprocessing
    # Set False for use limited memory
    BATCH = 5000  # After BATCH images, lmdb commits, if read_all_imgs = False
    n_thread = 40
    ########################################################
    img_folder = opt['img_folder']
    lmdb_save_path = opt['lmdb_save_path']
    meta_info = {'name': opt['name']}
    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with \'lmdb\'.")
    if osp.exists(lmdb_save_path):
        print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
        sys.exit(1)

    #### read all the image paths to a list
    print('Reading image path list ...')
    all_img_list = sorted(glob.glob(osp.join(img_folder, '*')))
    keys = []
    for img_path in all_img_list:
        keys.append(osp.splitext(osp.basename(img_path))[0])

    if read_all_imgs:
        #### read all images to memory (multiprocessing)
        dataset = {}  # store all image data. list cannot keep the order, use dict
        print('Read images with multiprocessing, #thread: {} ...'.format(n_thread))
        pbar = util.ProgressBar(len(all_img_list))

        def mycallback(arg):
            '''get the image data and update pbar'''
            key = arg[0]
            dataset[key] = arg[1]
            pbar.update('Reading {}'.format(key))

        pool = Pool(n_thread)
        for path, key in zip(all_img_list, keys):
            pool.apply_async(read_image_worker, args=(path, key), callback=mycallback)
        pool.close()
        pool.join()
        print('Finish reading {} images.\nWrite lmdb...'.format(len(all_img_list)))

    #### create lmdb environment
    data_size_per_img = cv2.imread(all_img_list[0], cv2.IMREAD_UNCHANGED).nbytes
    print('data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len(all_img_list)
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)

    #### write data to lmdb
    pbar = util.ProgressBar(len(all_img_list))
    txn = env.begin(write=True)
    resolutions = []
    for idx, (path, key) in enumerate(zip(all_img_list, keys)):
        pbar.update('Write {}'.format(key))
        key_byte = key.encode('ascii')
        data = dataset[key] if read_all_imgs else cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if data.ndim == 2:
            H, W = data.shape
            C = 1
        else:
            H, W, C = data.shape
        txn.put(key_byte, data)
        resolutions.append('{:d}_{:d}_{:d}'.format(C, H, W))
        if not read_all_imgs and idx % BATCH == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    env.close()
    print('Finish writing lmdb.')

    #### create meta information
    # check whether all the images are the same size
    assert len(keys) == len(resolutions)
    if len(set(resolutions)) <= 1:
        meta_info['resolution'] = [resolutions[0]]
        meta_info['keys'] = keys
        print('All images have the same resolution. Simplify the meta info.')
    else:
        meta_info['resolution'] = resolutions
        meta_info['keys'] = keys
        print('Not all images have the same resolution. Save meta info for each image.')

    pickle.dump(meta_info, open(osp.join(lmdb_save_path, 'meta_info.pkl'), "wb"))
    print('Finish creating lmdb meta info.')

def vimeo90k(mode):
    """Create lmdb for the Vimeo90K dataset, each image with a fixed size
    GT: [3, 256, 448]
        Now only need the 4th frame, e.g., 00001_0001_4
    LR: [3, 64, 112]
        1st - 7th frames, e.g., 00001_0001_1, ..., 00001_0001_7
    key:
        Use the folder and subfolder names, w/o the frame index, e.g., 00001_0001

    flow: downsampled flow: [3, 360, 320], keys: 00001_0001_4_[p3, p2, p1, n1, n2, n3]
        Each flow is calculated with GT images by PWCNet and then downsampled by 1/4
        Flow map is quantized by mmcv and saved in png format
    """
    #### configurations
    read_all_imgs = False  # whether real all images to memory with multiprocessing
    # Set False for use limited memory
    BATCH = 5000  # After BATCH images, lmdb commits, if read_all_imgs = False
    if mode == 'GT':
        img_folder = '../../datasets/vimeo90k/vimeo_septuplet/sequences'
        lmdb_save_path = '../../datasets/vimeo90k/vimeo90k_train_GT.lmdb'
        txt_file = '../../datasets/vimeo90k/vimeo_septuplet/sep_trainlist.txt'
        H_dst, W_dst = 256, 448
    elif mode == 'LR':
        img_folder = '../../datasets/vimeo90k/vimeo_septuplet_matlabLRx4/sequences'
        lmdb_save_path = '../../datasets/vimeo90k/vimeo90k_train_LR7frames.lmdb'
        txt_file = '../../datasets/vimeo90k/vimeo_septuplet/sep_trainlist.txt'
        H_dst, W_dst = 64, 112
    elif mode == 'flow':
        img_folder = '../../datasets/vimeo90k/vimeo_septuplet/sequences_flowx4'
        lmdb_save_path = '../../datasets/vimeo90k/vimeo90k_train_flowx4.lmdb'
        txt_file = '../../datasets/vimeo90k/vimeo_septuplet/sep_trainlist.txt'
        H_dst, W_dst = 128, 112
    else:
        raise ValueError('Wrong dataset mode: {}'.format(mode))
    n_thread = 40
    ########################################################
    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with \'lmdb\'.")
    if osp.exists(lmdb_save_path):
        print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
        sys.exit(1)

    #### read all the image paths to a list
    print('Reading image path list ...')
    with open(txt_file) as f:
        train_l = f.readlines()
        train_l = [v.strip() for v in train_l]
    all_img_list = []
    keys = []
    for line in train_l:
        folder = line.split('/')[0]
        sub_folder = line.split('/')[1]
        all_img_list.extend(glob.glob(osp.join(img_folder, folder, sub_folder, '*')))
        if mode == 'flow':
            for j in range(1, 4):
                keys.append('{}_{}_4_n{}'.format(folder, sub_folder, j))
                keys.append('{}_{}_4_p{}'.format(folder, sub_folder, j))
        else:
            for j in range(7):
                keys.append('{}_{}_{}'.format(folder, sub_folder, j + 1))
    all_img_list = sorted(all_img_list)
    keys = sorted(keys)
    if mode == 'GT':  # only read the 4th frame for the GT mode
        print('Only keep the 4th frame.')
        all_img_list = [v for v in all_img_list if v.endswith('im4.png')]
        keys = [v for v in keys if v.endswith('_4')]

    if read_all_imgs:
        #### read all images to memory (multiprocessing)
        dataset = {}  # store all image data. list cannot keep the order, use dict
        print('Read images with multiprocessing, #thread: {} ...'.format(n_thread))
        pbar = util.ProgressBar(len(all_img_list))

        def mycallback(arg):
            """get the image data and update pbar"""
            key = arg[0]
            dataset[key] = arg[1]
            pbar.update('Reading {}'.format(key))

        pool = Pool(n_thread)
        for path, key in zip(all_img_list, keys):
            pool.apply_async(read_image_worker, args=(path, key), callback=mycallback)
        pool.close()
        pool.join()
        print('Finish reading {} images.\nWrite lmdb...'.format(len(all_img_list)))

    #### write data to lmdb
    data_size_per_img = cv2.imread(all_img_list[0], cv2.IMREAD_UNCHANGED).nbytes
    print('data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len(all_img_list)
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)
    txn = env.begin(write=True)
    pbar = util.ProgressBar(len(all_img_list))
    for idx, (path, key) in enumerate(zip(all_img_list, keys)):
        pbar.update('Write {}'.format(key))
        key_byte = key.encode('ascii')
        data = dataset[key] if read_all_imgs else cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if 'flow' in mode:
            H, W = data.shape
            assert H == H_dst and W == W_dst, 'different shape.'
        else:
            H, W, C = data.shape
            assert H == H_dst and W == W_dst and C == 3, 'different shape.'
        txn.put(key_byte, data)
        if not read_all_imgs and idx % BATCH == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    env.close()
    print('Finish writing lmdb.')

    #### create meta information
    meta_info = {}
    if mode == 'GT':
        meta_info['name'] = 'Vimeo90K_train_GT'
    elif mode == 'LR':
        meta_info['name'] = 'Vimeo90K_train_LR'
    elif mode == 'flow':
        meta_info['name'] = 'Vimeo90K_train_flowx4'
    channel = 1 if 'flow' in mode else 3
    meta_info['resolution'] = '{}_{}_{}'.format(channel, H_dst, W_dst)
    key_set = set()
    for key in keys:
        if mode == 'flow':
            a, b, _, _ = key.split('_')
        else:
            a, b, _ = key.split('_')
        key_set.add('{}_{}'.format(a, b))
    meta_info['keys'] = list(key_set)
    pickle.dump(meta_info, open(osp.join(lmdb_save_path, 'meta_info.pkl'), "wb"))
    print('Finish creating lmdb meta info.')

def REDS(img_folder, lmdb_save_path, H_dst, W_dst, scale, read_all_imgs=False):
    """Create lmdb for the REDS dataset, each image with a fixed size
    GT: [3, 720, 1280],     key: 000_00000000
    LR: [3, H_dst, W_dst],  key: 000_00000000
    key: 000_00000000
    flow: downsampled flow: [3, 360, 320], keys: 000_00000005_[p2, p1, n1, n2]
        Each flow is calculated with the GT images by PWCNet and then downsampled by scale
        Flow map is quantized by mmcv and saved in png format
    """
    #### configurations
    BATCH = 5000  # After BATCH images, lmdb commits, if read_all_imgs = False
    n_thread = 40
    ########################################################
    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with \'lmdb\'.")
    if osp.exists(lmdb_save_path):
        print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
        sys.exit(1)

    #### read all the image paths to a list
    print('Reading image path list ...')
    # all_img_list = get_paths_from_images(img_folder)
    all_img_list = glob.glob(osp.join(img_folder, '*', '*'))
    all_img_list = sorted([x for x in all_img_list if is_image_file(x)])
    keys = []
    for img_path in all_img_list:
        folder = osp.basename(osp.dirname(img_path))
        # split_rlt = img_path.split('/')
        # folder = split_rlt[-2]
        # img_name = split_rlt[-1].split('.png')[0]
        img_name = osp.basename(img_path).split('.')[0]
        keys.append(folder + '_' + img_name)

    if read_all_imgs:
        #### read all images to memory (multiprocessing)
        dataset = {}  # store all image data. list cannot keep the order, use dict
        print('Read images with multiprocessing, #thread: {} ...'.format(n_thread))
        pbar = util.ProgressBar(len(all_img_list))

        def mycallback(arg):
            '''get the image data and update pbar'''
            key = arg[0]
            dataset[key] = arg[1]
            pbar.update('Reading {}'.format(key))

        pool = Pool(n_thread)
        for path, key in zip(all_img_list, keys):
            pool.apply_async(read_image_worker, args=(path, key), callback=mycallback)
        pool.close()
        pool.join()
        print('Finish reading {} images.\nWrite lmdb...'.format(len(all_img_list)))

    #### create lmdb environment
    data_size_per_img = cv2.imread(all_img_list[0], cv2.IMREAD_UNCHANGED).nbytes
    print('data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len(all_img_list)
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)

    #### write data to lmdb
    pbar = util.ProgressBar(len(all_img_list))
    txn = env.begin(write=True)
    for idx, (path, key) in enumerate(zip(all_img_list, keys)):
        pbar.update('Write {}'.format(key))
        key_byte = key.encode('ascii')
        data = dataset[key] if read_all_imgs else cv2.imread(path, cv2.IMREAD_UNCHANGED)
        H, W, C = data.shape
        assert H == H_dst and W == W_dst and C == 3, 'different shape.'
        txn.put(key_byte, data)
        if not read_all_imgs and idx % BATCH == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    env.close()
    print('Finish writing lmdb.')

    #### create meta information
    meta_info = {}
    meta_info['name'] = 'REDS_X{:.02f}_wval'.format(scale)
    channel = 3
    meta_info['resolution'] = '{}_{}_{}'.format(channel, H_dst, W_dst)
    meta_info['keys'] = keys
    pickle.dump(meta_info, open(osp.join(lmdb_save_path, 'meta_info.pkl'), "wb"))
    print('Finish creating lmdb meta info.')


def read_imgs_multi_thread(imgs, keys, n_thread=40):
    #### read all images to memory (multiprocessing)
    dataset = {}  # store all image data. list cannot keep the order, use dict
    print('Read images with multiprocessing, #thread: {} ...'.format(n_thread))
    pbar = util.ProgressBar(len(imgs))

    def mycallback(arg):
        '''get the image data and update pbar'''
        key = arg[0]
        dataset[key] = arg[1]
        pbar.update('Reading {}'.format(key))

    pool = Pool(n_thread)
    for path, key in zip(imgs, keys):
        pool.apply_async(read_image_worker, args=(path, key), callback=mycallback)
    pool.close()
    pool.join()
    print('Finish reading {} images.'.format(len(imgs)))
    return dataset


def MultiScaleREDS(img_root, lmdb_save_path, scales):
    """Create lmdb for the REDS dataset with multiple scales
    """
    #### configurations
    BATCH = 5000  # After BATCH images, lmdb commits, if read_all_imgs = False
    n_thread = 40
    ########################################################
    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with \'lmdb\'.")
    if osp.exists(lmdb_save_path):
        print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
        sys.exit(1)

    print('Reading image path list ...')
    # all_img_list = get_paths_from_images(img_folder)
    scale_folders = sorted(os.listdir(img_root))
    all_imgs, all_keys = [], []
    resolution = {}
    for i, folder in enumerate(scale_folders):
        print('[{:02d}/{:02d}] Reading scale-folder: {:s} ...'.format(i, 
                    len(scale_folders), folder))
        folder_dir = osp.join(img_root, folder)
        sub_folders = sorted(os.listdir(folder_dir))
        for sub in sub_folders:
            sub_dir = osp.join(folder_dir, sub)
            img_names  = sorted(os.listdir(sub_dir))
            imgs = [osp.join(sub_dir, name) for name in img_names]
            keys = [folder + '_' + sub + '_' + name[:-4] for name in img_names]
            all_imgs.extend(imgs)
            all_keys.extend(keys)
        resolution[folder] = cv2.imread(imgs[-1]).shape
    
    #### create lmdb environment
    data_size_per_img = cv2.imread(all_imgs[0], cv2.IMREAD_UNCHANGED).nbytes
    print('max data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len(all_imgs)
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)

    #### write data to lmdb
    txn = env.begin(write=True)
    for i in range(0, len(all_imgs), BATCH):
        imgs = all_imgs[i:i+BATCH]
        keys = all_keys[i:i+BATCH]
        batch_data = read_imgs_multi_thread(imgs, keys, n_thread)
        pbar = util.ProgressBar(len(imgs))
        for k, v in batch_data.items():
            pbar.update('Write {}'.format(k))
            key_byte = k.encode('ascii')
            txn.put(key_byte, v)
        txn.commit()
        txn = env.begin(write=True)
    txn.commit()
    env.close()
    print('Finish writing lmdb.')

    #### create meta information
    meta_info = {}
    meta_info['name'] = 'REDS_X1_X6_wval'
    channel = 3
    meta_info['resolution'] = resolution
    meta_info['keys'] = all_keys
    pickle.dump(meta_info, open(osp.join(lmdb_save_path, 'meta_info.pkl'), "wb"))
    print('Finish creating lmdb meta info.')

def test_lmdb_multi_scale(dataroot, dataset='REDS'):
    env = lmdb.open(dataroot, readonly=True, lock=False, readahead=False, meminit=False)
    meta_info = pickle.load(open(osp.join(dataroot, 'meta_info.pkl'), "rb"))
    print('Name: ', meta_info['name'])
    resolution = meta_info['resolution']
    keys = meta_info['keys']
    print('Resolution: ', meta_info['resolution'])
    print('# keys: ', len(meta_info['keys']))
    test_key = keys[0]
    with env.begin(write=False) as txn:
        buf = txn.get(test_key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    H, W, C = resolution[test_key.split('_')[0]]
    img = img_flat.reshape(H, W, C)
    cv2.imwrite('test.png', img)

def test_lmdb(dataroot, dataset='REDS'):
    env = lmdb.open(dataroot, readonly=True, lock=False, readahead=False, meminit=False)
    meta_info = pickle.load(open(osp.join(dataroot, 'meta_info.pkl'), "rb"))
    print('Name: ', meta_info['name'])
    print('Resolution: ', meta_info['resolution'])
    print('# keys: ', len(meta_info['keys']))
    # read one image
    if dataset == 'vimeo90k':
        key = '00001_0001_4'
    else:
        key = '000_00000000'
    print('Reading {} for test.'.format(key))
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    C, H, W = [int(s) for s in meta_info['resolution'].split('_')]
    img = img_flat.reshape(H, W, C)
    cv2.imwrite('test.png', img)


def parse_args():
    parser = argparse.ArgumentParser(description='Convert images to LMDB file for different scales')
    parser.add_argument('--src_dir', dest='src_dir', type=str,   default="../../datasets/REDS/LR_bicubic")
    parser.add_argument('--dst_dir', dest='dst_dir', type=str,   default="../../datasets/REDS/LR_x1_x6.lmdb")
    parser.add_argument('--dataset', dest='dataset', type=str,   default='REDS')
    parser.add_argument('--scale',   dest='scale',   type=float, default=-1)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    cfgs = parse_args()
    if cfgs.scale == -1:
        cfgs.scale = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, \
                      2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, \
                      3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, \
                      4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, \
                      5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0]
    # MultiScaleREDS(cfgs.src_dir, cfgs.dst_dir, cfgs.scale)
    test_lmdb_multi_scale(cfgs.dst_dir)
