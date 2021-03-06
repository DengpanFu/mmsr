import os
import os.path as osp
import logging
import yaml
import numpy as np
from utils.util import OrderedYaml
Loader, Dumper = OrderedYaml()


def parse(opt_path, opt_list=None, is_train=True):
    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)
    # export CUDA_VISIBLE_DEVICES
    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    opt['is_train'] = is_train
    set_default_opt(opt, 'no_log', False)
    set_default_opt(opt, 'auto_resume', True)
    set_default_opt(opt['train'], 'reduction', 'sum')
    set_default_opt(opt['train'], 'fix_edvr', False)
    set_default_opt(opt['datasets']['train'], 'sample_base', 'GT')

    if isinstance(opt['scale'], (int, float)) and opt['scale'] < -1:
        opt['scale'] = format_float(np.arange(1.1, -opt['scale']+ 0.05, 0.1).tolist())
    if isinstance(opt['scale'], (list, tuple)) and len(opt['scale']) > 0:
        refine_scales(opt)

    if opt['distortion'] == 'sr':
        scale = opt['scale']

    # datasets
    for phase, dataset in opt['datasets'].items():
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if opt['distortion'] == 'sr':
            # dataset['scale'] = scale
            set_default_opt(dataset, 'scale', scale)
        is_lmdb = False
        if dataset.get('dataroot_GT', None) is not None:
            dataset['dataroot_GT'] = osp.expanduser(dataset['dataroot_GT'])
            if dataset['dataroot_GT'].endswith('lmdb'):
                is_lmdb = True
        if dataset.get('dataroot_LQ', None) is not None:
            dataset['dataroot_LQ'] = osp.expanduser(dataset['dataroot_LQ'])
            if dataset['dataroot_LQ'].endswith('lmdb'):
                is_lmdb = True
        dataset['data_type'] = 'lmdb' if is_lmdb else 'img'
        if dataset['mode'].endswith('mc'):  # for memcached
            dataset['data_type'] = 'mc'
            dataset['mode'] = dataset['mode'].replace('_mc', '')

    # path
    for key, path in opt['path'].items():
        if path and key in opt['path'] and key != 'strict_load':
            opt['path'][key] = osp.expanduser(path)
    if not 'root' in opt['path']:
        opt['path']['root'] = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
    if is_train:
        if not 'experiments_root' in opt['path']:
            experiments_root = osp.join(opt['path']['root'], 'experiments', opt['name'])
            set_default_opt(opt['path'], 'experiments_root', experiments_root)
        else:
            experiments_root = opt['path']['experiments_root']
        set_default_opt(opt['path'], 'models', osp.join(experiments_root, 'models'))
        set_default_opt(opt['path'], 'training_state', osp.join(experiments_root, 'training_state'))
        set_default_opt(opt['path'], 'log', experiments_root)
        set_default_opt(opt['path'], 'val_images', osp.join(experiments_root, 'val_images'))
        set_default_opt(opt['path'], 'tb_logger', osp.join(experiments_root, 'tb_logger'))

        # change some options for debug mode
        if 'debug' in opt['name']:
            set_default_opt(opt['train'], 'val_freq', 8)
            set_default_opt(opt['logger'], 'print_freq', 1)
            set_default_opt(opt['logger'], 'save_checkpoint_freq', 8)
    else:  # test
        if not 'results_root' in opt['path']:
            results_root = osp.join(opt['path']['root'], 'results', opt['name'])
            set_default_opt(opt['path'], 'results_root', results_root)
        else:
            results_root = opt['path']['results_root']
        set_default_opt(opt['path'], 'log', results_root)

    # network
    if opt['distortion'] == 'sr':
        opt['network_G']['scale'] = scale
    # set some default network config for network G
    set_default_opt(opt['network_G'], 'which_model_G', 'MultiEDVR')
    set_default_opt(opt['network_G'], 'nf', 64)
    set_default_opt(opt['network_G'], 'nframes', 5)
    set_default_opt(opt['network_G'], 'groups', 8)
    set_default_opt(opt['network_G'], 'front_RBs', 5)
    set_default_opt(opt['network_G'], 'back_RBs', 10)
    set_default_opt(opt['network_G'], 'predeblur', False)
    set_default_opt(opt['network_G'], 'HR_in', False)
    set_default_opt(opt['network_G'], 'w_TSA', False)
    set_default_opt(opt['network_G'], 'ret_valid', False)
    set_default_opt(opt['network_G'], 'deform_lr_mult', 1.)
    # set some default network config for network D
    if 'network_D' in opt:
        set_default_opt(opt['network_D'], 'which_model_D', 'DCN3D')
        set_default_opt(opt['network_D'], 'input_nc', 3)
        set_default_opt(opt['network_D'], 'ndf', 64)
        set_default_opt(opt['network_D'], 'n_layers', 3)
        set_default_opt(opt['network_D'], 'num_d', 1)
        set_default_opt(opt['network_D'], 'use_sigmoid', False)
        set_default_opt(opt['network_D'], 'get_inter_feat', False)
        set_default_opt(opt['network_D'], 'has_bias', False)
        set_default_opt(opt['network_D'], 'has_sn', False)
        set_default_opt(opt['network_D'], 'max_ndf', 256)
        set_default_opt(opt['network_D'], 'conv_type', 'deform')

        # if has network_D, different optimizer/lr_scheduler options need
        set_default_opt(opt['train'], 'lr_G', 1e-4)
        set_default_opt(opt['train'], 'lr_D', 4e-4)
        set_default_opt(opt['train'], 'beta1_G', 0.9)
        set_default_opt(opt['train'], 'beta2_G', 0.99)
        set_default_opt(opt['train'], 'beta1_D', 0.9)
        set_default_opt(opt['train'], 'beta2_D', 0.99)
        set_default_opt(opt['train'], 'lr_scheme_G', 'StepLR')
        set_default_opt(opt['train'], 'lr_scheme_D', 'StepLR')
        set_default_opt(opt['train'], 'lr_step_G', 50000)
        set_default_opt(opt['train'], 'lr_step_D', 50000)
        set_default_opt(opt['train'], 'lr_gamma_G', 0.5)
        set_default_opt(opt['train'], 'lr_gamma_D', 0.5)
        set_default_opt(opt['train'], 'niter', 200000)
        set_default_opt(opt['train'], 'warmup_iter', -1)
        set_default_opt(opt['train'], 'restart_weights_G', [1, 1, 1])
        set_default_opt(opt['train'], 'restart_weights_D', [1, 1, 1])
        set_default_opt(opt['train'], 'eta_min_G', 1e-7)
        set_default_opt(opt['train'], 'eta_min_D', 1e-7)
        set_default_opt(opt['train'], 'gan_type', 'lsgan')
        set_default_opt(opt['train'], 'gan_weight', 1.0)

        # lr_scheme_G: CosineAnnealingLR_Restart
        # T_period_G: [150000, 150000, 150000, 150000]
        # restarts_G: [150000, 300000, 450000]
    set_default_opt(opt['train'], 'pixel_criterion', 'cb')
    set_default_opt(opt['train'], 'pixel_weight', 1.0)
    set_default_opt(opt['train'], 'val_freq', 5e3)
    set_default_opt(opt['train'], 'flow_weight', 0.3)

    if opt_list is not None:
        opt_from_list(opt, opt_list)
    return opt

def set_default_opt(opt, key, value):
    if not key in opt:
        opt[key] = value

def format_float(input, prec=1):
    format_str = lambda x: float("{:.{:d}f}".format(x, prec))
    if isinstance(input, (list, tuple)):
        return [format_str(x) for x in input]
    else:
        return format_str(input)

def refine_scales(opt):
    sample_base = opt['datasets']['train']['sample_base']
    if sample_base == 'GT':
        out_size = opt['datasets']['train'].get('GT_size', None)
        if out_size is None: 
            out_size = 256
            opt['datasets']['train']['GT_size'] = 256
        step = format_float(0.5 / out_size, 3)
        new_scales, sample_sizes = [], []
        for scale in opt['scale']:
            sample_size = int(np.round(out_size / scale))
            tmp_scale = format_float(out_size / sample_size, 3)
            tmp_size = int(sample_size * tmp_scale)
            while tmp_size > out_size:
                tmp_scale = format_float(tmp_scale - step, 3)
                tmp_size = int(sample_size * tmp_scale)
            while tmp_size < out_size:
                tmp_scale = format_float(tmp_scale + step, 3)
                tmp_size = int(sample_size * tmp_scale)
            new_scales.append(tmp_scale)
            sample_sizes.append(sample_size)
        opt['scale'] = new_scales
        opt['datasets']['train']['LQ_size'] = sample_sizes
    else:
        in_size = opt['datasets']['train'].get('LQ_size', None)
        if in_size is None: 
            in_size = 64
            opt['datasets']['train']['LQ_size'] = 64
        new_scales, sample_sizes = [], []
        for scale in opt['scale']:
            sample_size = int(np.round(scale * in_size))
            sample_sizes.append(sample_size)
            new_scale = sample_size / in_size
            if format_float(new_scale, 3) < new_scale:
                new_scale = format_float(new_scale + 0.001, 3)
            else:
                new_scale = format_float(new_scale, 3)
            new_scales.append(new_scale)
        opt['scale'] = new_scales
        opt['datasets']['train']['GT_size'] = sample_sizes


def opt_from_list(opt, opt_list):
    from ast import literal_eval
    assert(len(opt_list) % 2 == 0)
    for k, v in zip(opt_list[0::2], opt_list[1::2]):
        key_list = k.split('.')
        d = opt
        for subkey in key_list[:-1]:
            if subkey in d:
                d = d[subkey]
            else:
                d[subkey] = {}
                d = d[subkey]
        try:
            value = literal_eval(v)
        except:
            value = v
        d[key_list[-1]] = value

def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg

class NoneDict(dict):
    def __missing__(self, key):
        return None

# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt

def check_resume(opt, resume_iter):
    '''Check resume states and pretrain_model paths'''
    logger = logging.getLogger('base')
    if opt['path']['resume_state']:
        if opt['path'].get('pretrain_model_G', None) is not None or opt['path'].get(
                'pretrain_model_D', None) is not None:
            logger.warning('pretrain_model path will be ignored when resuming training.')

        opt['path']['pretrain_model_G'] = osp.join(opt['path']['models'],
                                                   '{}_G.pth'.format(resume_iter))
        logger.info('Set [pretrain_model_G] to ' + opt['path']['pretrain_model_G'])
        if 'gan' in opt['model'] or '3d_D' in opt['model']:
            opt['path']['pretrain_model_D'] = osp.join(opt['path']['models'],
                                                       '{}_D.pth'.format(resume_iter))
            logger.info('Set [pretrain_model_D] to ' + opt['path']['pretrain_model_D'])
