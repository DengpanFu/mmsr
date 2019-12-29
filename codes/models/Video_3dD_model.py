import logging
from collections import OrderedDict
import math
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss, GANLoss

logger = logging.getLogger('base')


class Videl3dDModel(BaseModel):
    def __init__(self, opt):
        super(Videl3dDModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, 
                device_ids=[torch.cuda.current_device()], find_unused_parameters=True)
        else:
            self.netG = DataParallel(self.netG)

        if self.is_train:
            self.netD = networks.define_D(opt).to(self.device)
            if opt['dist']:
                self.netG = DistributedDataParallel(self.netD, 
                    device_ids=[torch.cuda.current_device()], find_unused_parameters=True)
            else:
                self.netD = DataParallel(self.netD)

            self.netG.train()
            self.netD.train()

            ######## losses ########
            self.criterion_G, self.loss_mult_G = self.get_criterion(mode='pix', opt=train_opt)
            self.criterion_D, self.loss_mult_D = self.get_criterion(mode='gan', opt=train_opt)
            ######## optimizers ########
            self.optimizer_G = self.get_optimizer(self.netG, train_opt['lr_G'], 
                    train_opt['weight_decay_G'], train_opt['beta1_G'], train_opt['beta2_G'])
            self.optimizer_D = self.get_optimizer(self.netD, train_opt['lr_D'], 
                    train_opt['weight_decay_D'], train_opt['beta1_D'], train_opt['beta2_D'])
            self.optimizers = [self.optimizer_G, self.optimizer_D]

            ######## schedulers ########
            self.scheduler_G = self.get_lr_scheduler(self.optimizer_G, train_opt['lr_scheme_G'], 
                                    train_opt, mode='G')
            self.scheduler_D = self.get_lr_scheduler(self.optimizer_D, train_opt['lr_scheme_D'], 
                                    train_opt, mode='D')
            self.schedulers = [self.scheduler_G, self.scheduler_D]

            self.log_dict = OrderedDict()

        self.n_gpus = len(self.opt['gpu_ids'])
        # print network
        # self.print_network()
        self.load()

    def get_criterion(self, mode, opt):
        if mode == 'pix':
            loss_type = opt['pixel_criterion']
            if loss_type == 'l1':
                criterion = nn.L1Loss(reduction='sum').to(self.device)
            elif loss_type == 'l2':
                criterion = nn.MSELoss(reduction='sum').to(self.device)
            elif loss_type == 'cb':
                criterion = CharbonnierLoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized for pixel'.
                    format(loss_type))
            weight = opt['pixel_weight']
        elif mode == 'gan':
            criterion = GANLoss(opt['gan_type'], 1.0, 0.0).to(self.device)
            weight = opt['gan_weight']
        else:
            raise TypeError('Unknown type: {} for criterion'.format(mode))
        return criterion, weight

    def get_optimizer(self, net, lr, wd, beta1, beta2):
        if wd is None: wd = 0
        if beta1 is None: beta1 = 0.9
        if beta2 is None: beta2 = 0.99
        optim_params = []
        for k, v in net.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                if self.rank <= 0:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
        optimizer = torch.optim.Adam(optim_params, lr=lr, weight_decay=wd,
                                     betas=(beta1, beta2))
        return optimizer

    def get_lr_scheduler(self, optimizer, lr_scheme, opt, mode='G'):
        assert mode in ['G', 'D']
        if lr_scheme == 'MultiStepLR_Restart':
            scheduler = lr_scheduler.MultiStepLR_Restart(optimizer, opt['lr_steps_'+mode],
                        restarts=opt['restarts_'+mode], weights=opt['restart_weights_'+mode],
                        gamma=opt['lr_gamma_'+mode], clear_state=opt['clear_state_'+mode])
        elif lr_scheme == 'CosineAnnealingLR_Restart':
            scheduler = lr_scheduler.CosineAnnealingLR_Restart(optimizer, opt['T_period_'+mode], 
                        eta_min=opt['eta_min_'+mode], restarts=opt['restarts_'+mode], 
                        weights=opt['restart_weights_'+mode])
        elif lr_scheme == 'StepLR':
            scheduler = lr_scheduler.StepLR(optimizer, 
                        step_size=opt['lr_step_'+mode], gamma=opt['lr_gamma_'+mode])
        elif lr_scheme == 'MultiStepLR':
            scheduler = lr_scheduler.MultiStepLR(optimizer, 
                        milestones=opt['lr_steps_'+mode], gamma=opt['lr_gamma_'+mode])
        else:
            raise TypeError('Unknown lr scheduler type: {}'.format(lr_scheme))
        return scheduler

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQs'].to(self.device)
        if need_GT:
            self.real_H = data['GT'].to(self.device)

    def set_requires_grad(self, net, requires_grad=False):
        for param in net.parameters():
            param.requires_grad = requires_grad

    def backward_G(self):
        self.loss_pix = self.criterion_G(self.fake_H, self.real_H)
        self.pred_fake_H = self.netD(self.fake_H)
        self.loss_G_fake = self.criterion_D(self.pred_fake_H[0], True)
        self.loss_G = self.loss_mult_G * self.loss_pix + self.loss_mult_D * self.loss_G_fake
        self.loss_G.backward()
        self.optimizer_G.step()

    def backward_D(self):
        self.pred_real_H = self.netD(self.real_H)
        self.pred_fake_H = self.netD(self.fake_H.detach())
        self.loss_D_real = self.criterion_D(self.pred_real_H[0], True)
        self.loss_D_fake = self.criterion_D(self.pred_fake_H[0], False)
        self.loss_D = (self.loss_D_real + self.loss_D_fake) / 2
        self.loss_D.backward()
        self.optimizer_D.step()

    def optimize_parameters(self, step):
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L)
        self.backward_G()

        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def get_current_log(self):
        # set log
        self.log_dict['l_pix'] = self.loss_pix.item()
        self.log_dict['l_G_f'] = self.loss_G_fake.item()
        self.log_dict['l_G'] = self.loss_G.item()
        self.log_dict['l_D_r'] = self.loss_D_real.item()
        self.log_dict['l_D_f'] = self.loss_G_fake.item()
        self.log_dict['l_D'] = self.loss_D.item()
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)
        if self.is_train:
            # Discriminator
            s, n = self.get_network_description(self.netD)
            if isinstance(self.netD, nn.DataParallel) or isinstance(self.netD,
                                                                    DistributedDataParallel):
                net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
                                                 self.netD.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.netD.__class__.__name__)
            if self.rank <= 0:
                logger.info('Network D structure: {}, with parameters: {:,d}'.format(
                    net_struc_str, n))
                logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
        load_path_D = self.opt['path']['pretrain_model_D']
        if self.opt['is_train'] and load_path_D is not None:
            logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD, self.opt['path']['strict_load'])

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
        self.save_network(self.netD, 'D', iter_step)

