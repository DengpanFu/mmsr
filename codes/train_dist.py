import os
import math
import argparse
import random
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler, DistMetaIterSampler

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model


def init_dist(backend='nccl', **kwargs):
    """initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default='options/train/Meta_EDVR_M_woTSA.yml', 
                        help='Path to option YAML file.')
    parser.add_argument('--set', dest='set_opt', default=None, nargs=argparse.REMAINDER, 
                        help='set options')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='pytorch',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, args.set_opt, is_train=True)

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        print('Training from state: {}'.format(opt['path']['resume_state']))
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    elif opt['auto_resume']:
        exp_dir = opt['path']['experiments_root']
        # first time run: create dirs
        if not os.path.exists(exp_dir):
            resume_state = None
            if rank <= 0:
                util.mkdir(exp_dir)
                util.mkdir(opt['path']['models'])
                util.mkdir(opt['path']['training_state'])
                util.mkdir(opt['path']['val_images'])
                util.mkdir(opt['path']['tb_logger'])
        else:
            if rank <= 0:
                util.mkdir(opt['path']['models'])
                util.mkdir(opt['path']['training_state'])
                util.mkdir(opt['path']['val_images'])
                util.mkdir(opt['path']['tb_logger'])
            # detect experiment directory and get the latest state
            state_dir = opt['path']['training_state']
            state_files = [x for x in os.listdir(state_dir) if x.endswith('state')]
            # no valid state detected
            if len(state_files) < 1:
                print('No previous training state found, train from start state')
                resume_state = None
            else:
                state_files = sorted(state_files, key=lambda x: int(x.split('.')[0]))
                latest_state = state_files[-1]
                print('Training from lastest state: {}'.format(latest_state))
                latest_state_file = os.path.join(state_dir, latest_state)
                opt['path']['resume_state'] = latest_state_file
                device_id = torch.cuda.current_device()
                resume_state = torch.load(latest_state_file, 
                        map_location=lambda storage, loc: storage.cuda(device_id))
                option.check_resume(opt, resume_state['iter'])
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None and not opt['auto_resume'] and not opt['no_log']:
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
            tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
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
            total_epochs = int(math.ceil(total_iters / train_size))
            if dataset_opt['mode'] in ['MetaREDS', 'MetaREDSOnline', 'UPREDS', 'UPVimeo']:
                train_sampler = DistMetaIterSampler(train_set, world_size, rank, 
                    dataset_opt['batch_size'], len(opt['scale']), dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            elif dataset_opt['mode'] in ['REDS', 'MultiREDS', 'REDSImg', 'REDSMultiImg']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    #### create model
    model = create_model(opt)

    #### resume training
    used_data_need_skip = False
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
        used_data_iters = current_step - start_epoch * len(train_loader)
        if used_data_iters > 0:
            used_data_need_skip = True
        used_cnt = 0
    else:
        current_step = 0
        start_epoch = 0
        used_data_iters, used_cnt = 0, 0
    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            if used_data_need_skip:
                # resume training, skip some data that already explored.
                used_cnt += 1
                if used_cnt >= used_data_iters:
                    used_data_need_skip = False
            else:
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
                        if v is not None:
                            message += '{:s}: {:.4e} '.format(k, v)
                        # tensorboard logger
                        if opt['use_tb_logger'] and 'debug' not in opt['name']:
                            if rank <= 0 and v is not None:
                                tb_logger.add_scalar(k, v, current_step)
                    if rank <= 0:
                        logger.info(message)
                        if current_step % (total_iters // 500) == 0:
                            print("PROGRESS: {:02d}%".format(int(current_step/total_iters*100)))
                #### validation
                if opt['datasets'].get('val', None) and current_step % opt['train']['val_freq'] == 0:
                    # video restoration validation
                    psnr_rlt = {}  # with border and center frames
                    if rank == 0:
                        pbar = util.ProgressBar(len(val_set))
                    for idx in range(rank, len(val_set), world_size):
                        val_data = val_set[idx]
                        val_data['LQs'].unsqueeze_(0)
                        val_data['GT'].unsqueeze_(0)
                        folder = val_data['folder']
                        idx_d, max_idx = val_data['idx'].split('/')
                        idx_d, max_idx = int(idx_d), int(max_idx)
                        if psnr_rlt.get(folder, None) is None:
                            psnr_rlt[folder] = torch.zeros(max_idx, dtype=torch.float32,
                                                           device='cuda')
                        # tmp = torch.zeros(max_idx, dtype=torch.float32, device='cuda')
                        model.feed_data(val_data)
                        model.test()
                        visuals = model.get_current_visuals()
                        rlt_img = util.tensor2img(visuals['rlt'])  # uint8
                        gt_img = util.tensor2img(visuals['GT'])  # uint8
                        # calculate PSNR
                        psnr_rlt[folder][idx_d] = util.calculate_psnr(rlt_img, gt_img)
                        # print("{}-{}: {:.4f}".format(folder, idx_d, psnr_rlt[folder][idx_d]))
                        # exit()

                        if rank == 0:
                            for _ in range(world_size):
                                pbar.update('Test {} - {}/{}'.format(folder, idx_d, max_idx))
                    # # collect data
                    for _, v in psnr_rlt.items():
                        dist.reduce(v, 0)
                    dist.barrier()

                    if rank == 0:
                        psnr_rlt_avg = {}
                        psnr_total_avg = 0.
                        for k, v in psnr_rlt.items():
                            psnr_rlt_avg[k] = torch.mean(v).cpu().item()
                            psnr_total_avg += psnr_rlt_avg[k]
                        psnr_total_avg /= len(psnr_rlt)
                        log_s = '# Validation # PSNR: {:.3f}:'.format(psnr_total_avg)
                        for k, v in psnr_rlt_avg.items():
                            log_s += ' {}: {:.3f}'.format(k, v)
                        logger.info(log_s)
                        if opt['use_tb_logger'] and 'debug' not in opt['name']:
                            tb_logger.add_scalar('psnr_avg', psnr_total_avg, current_step)
                            for k, v in psnr_rlt_avg.items():
                                tb_logger.add_scalar(k, v, current_step)

                #### save models and training states
                if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                    if rank <= 0:
                        logger.info('Saving models and training states.')
                        model.save(current_step)
                        model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')
        tb_logger.close()


if __name__ == '__main__':
    main()
