import os
import math
import argparse
import random
import logging

import torch
from data.data_sampler import MetaIterSampler, IterSampler

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default='options/train/train_EDVR_woTSA_M.yml', 
                        help='Path to option YAML file.')
    parser.add_argument('--set', dest='set_opt', default=None, nargs=argparse.REMAINDER, 
                        help='set options')
    args = parser.parse_args()
    opt = option.parse(args.opt, args.set_opt, is_train=True)

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    elif opt['auto_resume']:
        # detect experiment directory and get the latest state
        exp_dir = opt['path']['experiments_root']
    else:
        resume_state = None

    if resume_state is None and not opt['no_log']:
        util.mkdir_and_rename(
            opt['path']['experiments_root'])  # rename experiment folder if exists
        util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                     and 'pretrain_model' not in key and 'resume' not in key))

    # config loggers. Before it, the log will not work
    util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                      screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))
    # tensorboard logger
    if opt['use_tb_logger'] and 'debug' not in opt['name']:
        version = float(torch.__version__[0:3])
        if version >= 1.2:  # PyTorch 1.1
            from torch.utils.tensorboard import SummaryWriter
        else:
            logger.info(
                'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
            from tensorboardX import SummaryWriter
        tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    if opt['datasets']['train']['ratio']:
        dataset_ratio = opt['datasets']['train']['ratio']
    else:
        dataset_ratio = 200   # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            if dataset_opt['mode'] in ['MetaREDS', 'MetaREDSOnline']:
                train_sampler = MetaIterSampler(train_set, 
                    dataset_opt['batch_size'], len(opt['scale']), dataset_ratio)
            elif dataset_opt['mode'] == 'REDS':
                train_sampler = IterSampler(train_set, dataset_opt['batch_size'], dataset_ratio)
            else:
                train_sampler = None

            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                len(train_set), train_size))
            logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            logger.info('Number of val images in [{:s}]: {:d}'.format(
                dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))

    #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            #### update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            #### training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)
            #### log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '[epoch:{:3d}, iter:{:8,d}, lr:('.format(epoch, current_step)
                for v in model.get_current_learning_rate():
                    message += '{:.3e},'.format(v)
                message += ')] '
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)
            #### validation
            if opt['datasets'].get('val', None) and current_step % opt['train']['val_freq'] == 0:
                pbar = util.ProgressBar(len(val_loader))
                psnr_rlt = {}  # with border and center frames
                psnr_rlt_avg = {}
                psnr_total_avg = 0.
                for val_data in val_loader:
                    folder = val_data['folder'][0]
                    idx_d = val_data['idx'].item()
                    # border = val_data['border'].item()
                    if psnr_rlt.get(folder, None) is None:
                        psnr_rlt[folder] = []

                    model.feed_data(val_data)
                    model.test()
                    visuals = model.get_current_visuals()
                    rlt_img = util.tensor2img(visuals['rlt'])  # uint8
                    gt_img = util.tensor2img(visuals['GT'])  # uint8

                    # calculate PSNR
                    psnr = util.calculate_psnr(rlt_img, gt_img)
                    psnr_rlt[folder].append(psnr)
                    pbar.update('Test {} - {}'.format(folder, idx_d))
                for k, v in psnr_rlt.items():
                    psnr_rlt_avg[k] = sum(v) / len(v)
                    psnr_total_avg += psnr_rlt_avg[k]
                psnr_total_avg /= len(psnr_rlt)
                log_s = '# Validation # PSNR: {:.4e}:'.format(psnr_total_avg)
                for k, v in psnr_rlt_avg.items():
                    log_s += ' {}: {:.4e}'.format(k, v)
                logger.info(log_s)
                if opt['use_tb_logger'] and 'debug' not in opt['name']:
                    tb_logger.add_scalar('psnr_avg', psnr_total_avg, current_step)
                    for k, v in psnr_rlt_avg.items():
                        tb_logger.add_scalar(k, v, current_step)

            #### save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(current_step)
                model.save_training_state(epoch, current_step)

    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of training.')
    tb_logger.close()


if __name__ == '__main__':
    main()
