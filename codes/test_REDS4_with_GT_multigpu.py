'''
Test Vid4 (SR) and REDS4 (SR-clean, SR-blur, deblur-clean, deblur-compression) datasets
'''

import os
import os.path as osp
import argparse, glob
import logging
import numpy as np
import cv2
import torch
from torch import nn
import utils.util as util
import data.util as data_util
import models.archs.EDVR_arch as EDVR_arch


def main(opts):
    ################## configurations #################
    device = torch.device('cuda')
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpus
    cache_all_imgs = opts.cache > 0
    n_gpus = len(opts.gpus.split(','))
    flip_test, save_imgs = False, False
    scale = 4
    N_in, nf = 5, 64
    back_RBs = 10
    w_TSA = False
    predeblur, HR_in = False, False
    crop_border = 0
    border_frame = N_in // 2
    padding = 'new_info'

    ################## model files ####################
    model_dir = opts.model_dir
    if osp.isfile(model_dir):
        model_names = [osp.basename(model_dir)]
        model_dir = osp.dirname(model_dir)
    elif osp.isdir(model_dir):
        model_names = [x for x in os.listdir(model_dir) if str.isdigit(x.split('_')[0])]
        model_names = sorted(model_names, key=lambda x:int(x.split("_")[0]))
    else:
        raise IOError('Invalid model_dir: {}'.format(model_dir))

    ################## dataset ########################
    test_subs = sorted(os.listdir(opts.test_dir))
    gt_subs = os.listdir(opts.gt_dir)
    valid_test_subs = [sub in gt_subs for sub in test_subs]
    assert(all(valid_test_subs)), 'Invalid sub folders exists in {}'.format(opts.test_dir)
    if cache_all_imgs:
        print('Cacheing all testing images ...')
        all_imgs = {}
        for sub in test_subs:
            print('Reading sub-folder: {} ...'.format(sub))
            test_sub_dir = osp.join(opts.test_dir, sub)
            gt_sub_dir = osp.join(opts.gt_dir, sub)
            all_imgs[sub] = {'test': [], 'gt': []}
            im_names = sorted(os.listdir(test_sub_dir))
            for i, name in enumerate(im_names):
                test_im_path = osp.join(test_sub_dir, name)
                gt_im_path = osp.join(gt_sub_dir, name)
                test_im = cv2.imread(test_im_path, cv2.IMREAD_UNCHANGED)[:,:,(2,1,0)]
                test_im = test_im.astype(np.float32).transpose((2,0,1)) / 255.
                all_imgs[sub]['test'].append(test_im)
                gt_im = cv2.imread(gt_im_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                all_imgs[sub]['gt'].append(gt_im)

    all_psnrs = []
    for model_name in model_names[-1:]:
        model_path = osp.join(model_dir, model_name)
        exp_name = model_name.split('_')[0]
        if 'meta' in opts.mode.lower():
            model = EDVR_arch.MetaEDVR(nf=nf, nframes=N_in, groups=8, front_RBs=5, 
                                       center=None, back_RBs=back_RBs, predeblur=predeblur, 
                                       HR_in=HR_in, w_TSA=w_TSA)
        elif opts.mode.lower() == 'edvr':
            model = EDVR_arch.EDVR(nf=nf, nframes=N_in, groups=8, front_RBs=5, 
                                   center=None, back_RBs=back_RBs, predeblur=predeblur, 
                                   HR_in=HR_in, w_TSA=w_TSA)
        else:
            raise TypeError('Unknown model mode: {}'.format(opts.mode))
        save_folder = osp.join(opts.save_dir, exp_name)
        util.mkdirs(save_folder)
        util.setup_logger(exp_name, save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
        logger = logging.getLogger(exp_name)

        #### log info
        logger.info('Data: {}'.format(opts.test_dir))
        logger.info('Padding mode: {}'.format(padding))
        logger.info('Model path: {}'.format(model_path))
        logger.info('Save images: {}'.format(save_imgs))
        logger.info('Flip test: {}'.format(flip_test))

        #### set up the models
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        if n_gpus > 1:
            model = nn.DataParallel(model)
        model = model.to(device)

        avg_psnrs, avg_psnr_centers, avg_psnr_borders = [], [], []
        evaled_subs = []

        # for each subfolder
        for sub in test_subs:
            evaled_subs.append(sub)
            test_sub_dir = osp.join(opts.test_dir, sub)
            gt_sub_dir = osp.join(opts.gt_dir, sub)
            img_names = sorted(os.listdir(test_sub_dir))
            max_idx = len(img_names)
            if save_imgs:
                save_subfolder = osp.join(save_folder, sub)
                util.mkdirs(save_subfolder)

            #### get LQ and GT images
            if not cache_all_imgs:
                img_LQs, img_GTs = [], []
                for i, name in enumerate(img_names):
                    test_im_path = osp.join(test_sub_dir, name)
                    gt_im_path = osp.join(gt_sub_dir, name)
                    test_im = cv2.imread(test_im_path, cv2.IMREAD_UNCHANGED)[:,:,(2,1,0)]
                    test_im = test_im.astype(np.float32).transpose((2,0,1)) / 255.
                    gt_im = cv2.imread(gt_im_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                    img_LQs.append(test_im)
                    img_GTs.append(gt_im)
            else:
                img_LQs = all_imgs[sub]['test']
                img_GTs = all_imgs[sub]['gt']

            avg_psnr, avg_psnr_border, avg_psnr_center, N_border, N_center = 0, 0, 0, 0, 0

            # process each image
            for i in range(0, max_idx, n_gpus):
                end = min(i + n_gpus, max_idx)
                select_idxs = [data_util.index_generation(j, max_idx, N_in, 
                            padding=padding) for j in range(i, end)]
                imgs = []
                for select_idx in select_idxs:
                    im = torch.from_numpy(np.stack([img_LQs[k] for k in select_idx]))
                    imgs.append(im)
                if (i + n_gpus) > max_idx:
                    for _ in range(max_idx, i + n_gpus):
                        imgs.append(torch.zeros_like(im))
                imgs = torch.stack(imgs, 0).to(device)

                if flip_test:
                    output = util.flipx4_forward(model, imgs)
                else:
                    if 'meta' in opts.mode.lower():
                        output = util.meta_single_forward(model, imgs, scale, n_gpus)
                    else:
                        output = util.single_forward(model, imgs)
                output = [util.tensor2img(x).astype(np.float32) for x in output]

                if save_imgs:
                    for ii in range(i, end):
                        cv2.imwrite(osp.join(save_subfolder, '{}.png'.format(
                                img_names[ii])), output[ii-i].astype(np.uint8))

                # calculate PSNR
                GT = np.copy(img_GTs[i:end])

                output = util.crop_border(output, crop_border)
                GT = util.crop_border(GT, crop_border)
                for m in range(i, end):
                    crt_psnr = util.calculate_psnr(output[m-i], GT[m-i])
                    logger.info('{:3d} - {:25} \tPSNR: {:.6f} dB'.format(m + 1, img_names[m], crt_psnr))

                    if m >= border_frame and m < max_idx - border_frame:  # center frames
                        avg_psnr_center += crt_psnr
                        N_center += 1
                    else:  # border frames
                        avg_psnr_border += crt_psnr
                        N_border += 1


            avg_psnr = (avg_psnr_center + avg_psnr_border) / (N_center + N_border)
            avg_psnr_center = avg_psnr_center / N_center
            avg_psnr_border = 0 if N_border == 0 else avg_psnr_border / N_border
            avg_psnrs.append(avg_psnr)
            avg_psnr_centers.append(avg_psnr_center)
            avg_psnr_borders.append(avg_psnr_border)

            logger.info('Folder {} - Average PSNR: {:.6f} dB for {} frames; '
                        'Center PSNR: {:.6f} dB for {} frames; '
                        'Border PSNR: {:.6f} dB for {} frames.'.format(sub, avg_psnr,
                                                                       (N_center + N_border),
                                                                       avg_psnr_center, N_center,
                                                                       avg_psnr_border, N_border))

        logger.info('################ Tidy Outputs ################')
        for subfolder_name, psnr, psnr_center, psnr_border in zip(evaled_subs, avg_psnrs,
                                                                  avg_psnr_centers, avg_psnr_borders):
            logger.info('Folder {} - Average PSNR: {:.6f} dB. '
                        'Center PSNR: {:.6f} dB. '
                        'Border PSNR: {:.6f} dB.'.format(subfolder_name, psnr, psnr_center,
                                                         psnr_border))
        logger.info('################ Final Results ################')
        logger.info('Data: {}'.format(opts.test_dir))
        logger.info('Padding mode: {}'.format(padding))
        logger.info('Model path: {}'.format(model_path))
        logger.info('Save images: {}'.format(save_imgs))
        logger.info('Flip test: {}'.format(flip_test))
        logger.info('Total Average PSNR: {:.6f} dB for {} clips. '
                    'Center PSNR: {:.6f} dB. Border PSNR: {:.6f} dB.'.format(
                        sum(avg_psnrs) / len(avg_psnrs), len(test_subs),
                        sum(avg_psnr_centers) / len(avg_psnr_centers),
                        sum(avg_psnr_borders) / len(avg_psnr_borders)))

        all_psnrs.append(avg_psnrs + [model_name])
    with open(osp.join(opts.save_dir, 'all_psnrs.txt'), 'w') as f:
        for psnrs in all_psnrs:
            f.write("{:>14s}: {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} \n".format(psnrs[-1], 
                sum(psnrs[:-1])/4., psnrs[0], psnrs[1], psnrs[2], psnrs[3]))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',       dest='mode',       type=str,  default='meta')
    parser.add_argument('--gpus',       dest='gpus',       type=str,  default='0')
    parser.add_argument('--cache',      dest='cache',      type=int,  default=1)
    parser.add_argument('--model_dir',  dest='model_dir',  type=str,  
                                        default='../experiments/MetaEDVRwoTSA_M_scratch/models')
    parser.add_argument('--gt_dir',     dest='gt_dir',     type=str,  
                                        default='../datasets/REDS/REDS4/GT')
    parser.add_argument('--test_dir',   dest='test_dir',   type=str,  
                                        default='../datasets/REDS/REDS4/sharp_bicubic')
    parser.add_argument('--save_dir',   dest='save_dir',   type=str,  
                                        default='../results/')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    opts = parse_args()
    main(opts)

