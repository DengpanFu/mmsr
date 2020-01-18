import logging
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss, MaskedCharbonnierLoss
# import models.archs.Flow_arch as Flow_arch

logger = logging.getLogger('base')


class VideoBaseModel(BaseModel):
    def __init__(self, opt):
        super(VideoBaseModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = self.get_network(opt, 'G')
        self.n_gpus = len(self.opt['gpu_ids'])

        if self.is_train:
            self.netG.train()

            ######## loss ########
            self.cri_pix, self.l_pix_w = self.get_criterion(mode='pix', opt=train_opt)
            ######## optimizers ########
            self.optimizer_G = self.get_optimizer(self.netG, opt=train_opt)
            self.optimizers.append(self.optimizer_G)
            ######## schedulers ########
            self.scheduler_G = self.get_lr_scheduler(self.optimizer_G, train_opt)
            self.schedulers = [self.scheduler_G]

            self.log_dict = OrderedDict()
        #### print network
        # self.print_network()
        self.load()

    def get_network(self, opt, mode='G'):
        assert(mode in ['G', 'D', 'F'])
        if mode == 'G':
            net = networks.define_G(opt).to(self.device)
        elif mode == 'D':
            net = networks.define_D(opt).to(self.device)
        elif mode == 'F':
            net = networks.define_F(opt).to(self.device)
        if opt['dist']:
            net = DistributedDataParallel(net, 
                        device_ids=[torch.cuda.current_device()], 
                        find_unused_parameters=True, 
                        broadcast_buffers=False)
        else:
            net = DataParallel(net)
        return net

    def get_criterion(self, mode, opt):
        if mode == 'pix':
            loss_type = opt['pixel_criterion']
            if loss_type == 'l1':
                criterion = nn.L1Loss(reduction=opt['reduction']).to(self.device)
            elif loss_type == 'l2':
                criterion = nn.MSELoss(reduction=opt['reduction']).to(self.device)
            elif loss_type == 'cb':
                criterion = CharbonnierLoss(reduction=opt['reduction']).to(self.device)
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

    def get_optimizer(self, net, opt):
        wd = opt['weight_decay_G'] if opt['weight_decay_G'] else 0
        beta1 = opt['beta1'] if opt['beta1'] else 0.9
        beta2 = opt['beta2'] if opt['beta2'] else 0.99
        if opt['ft_tsa_only']:
            normal_params, tsa_fusion_params = [], []
            for k, v in net.named_parameters():
                if v.requires_grad:
                    if 'tsa_fusion' in k:
                        tsa_fusion_params.append(v)
                    else:
                        normal_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            optim_params = [{'params': normal_params, 'lr': opt['lr_G'], 'name': 'normal'}, 
                            {'params': tsa_fusion_params, 'lr': opt['lr_G'], 'name': 'tsa'}]
        else:
            optim_params = []
            for k, v in net.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))

        optimizer = torch.optim.Adam(optim_params, lr=opt['lr_G'], weight_decay=wd,
                                     betas=(beta1, beta2))
        return optimizer

    def get_lr_scheduler(self, optimizer, opt, mode=None):
        assert mode in ['G', 'D', None]
        fpm = lambda x: x + '_' + mode if mode else x
        lr_scheme = opt['lr_scheme']
        if lr_scheme == 'MultiStepLR_Restart':
            scheduler = lr_scheduler.MultiStepLR_Restart(optimizer, opt[fpm('lr_steps')],
                        restarts=opt[fpm('restarts')], weights=opt[fpm('restart_weights')],
                        gamma=opt[fpm('lr_gamma')], clear_state=opt[fpm('clear_state')])
        elif lr_scheme == 'CosineAnnealingLR_Restart':
            scheduler = lr_scheduler.CosineAnnealingLR_Restart(optimizer, opt[fpm('T_period')], 
                        eta_min=opt[fpm('eta_min')], restarts=opt[fpm('restarts')], 
                        weights=opt[fpm('restart_weights')])
        elif lr_scheme == 'StepLR':
            scheduler = lr_scheduler.StepLR(optimizer, 
                        step_size=opt[fpm('lr_step')], gamma=opt[fpm('lr_gamma')])
        elif lr_scheme == 'MultiStepLR':
            scheduler = lr_scheduler.MultiStepLR(optimizer, 
                        milestones=opt[fpm('lr_steps')], gamma=opt[fpm('lr_gamma')])
        elif lr_scheme == 'CosineAnnealingLR':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 
                        T_max=opt[fpm('T_max')], eta_min=opt[fpm('eta_min')])
        else:
            raise TypeError('Unknown lr scheduler type: {}'.format(lr_scheme))
        return scheduler

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQs'].to(self.device)
        if need_GT:
            self.real_H = data['GT'].to(self.device)

    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0

    def optimize_parameters(self, step):
        if self.opt['train']['ft_tsa_only'] and step < self.opt['train']['ft_tsa_only']:
            self.set_params_lr_zero()

        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L)

        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        l_pix.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def get_current_log(self):
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

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)

class UPVideoModel(VideoBaseModel):
    def __init__(self, opt):
        self.ret_valid = opt['network_G']['ret_valid']
        self.deform_lr_mult = opt['network_G']['deform_lr_mult']
        super(UPVideoModel, self).__init__(opt)
        self.invalid_cnt = 0
        if self.rank <= 0:
            logger.info('Optimize deform_conv offset params with lr_mult: {}'.
                format(self.deform_lr_mult))
            if self.ret_valid:
                logger.info('Fliter out the deform conv offset mean > 100')

    def get_optimizer(self, net, opt):
        wd = opt['weight_decay_G'] if opt['weight_decay_G'] else 0
        beta1 = opt['beta1'] if opt['beta1'] else 0.9
        beta2 = opt['beta2'] if opt['beta2'] else 0.99
        if opt['ft_tsa_only']:
            normal_params, tsa_fusion_params, deform_params = [], [], []
            for k, v in net.named_parameters():
                if v.requires_grad:
                    if 'tsa_fusion' in k:
                        tsa_fusion_params.append(v)
                    elif 'conv_offset_mask' in k:
                        deform_params.append(v)
                    else:
                        normal_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            optim_params = [{'params': normal_params, 'lr': opt['lr_G'], 'name': 'normal'}, 
                            {'params': tsa_fusion_params, 'lr': opt['lr_G'], 'name': 'tsa'}, 
                            {'params': deform_params, 'name': 'deform_offset_params', 
                                    'lr': opt['lr_G'] * self.deform_lr_mult}]
        else:
            normal_params, deform_params = [], []
            optim_params = []
            for k, v in net.named_parameters():
                if v.requires_grad:
                    if 'conv_offset_mask' in k:
                        deform_params.append(v)
                    else:
                        normal_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            optim_params = [{'params': normal_params, 'lr': opt['lr_G'], 'name': 'normal'}, 
                            {'params': deform_params, 'name': 'deform_offset_params', 
                                    'lr': opt['lr_G'] * self.deform_lr_mult}]

        optimizer = torch.optim.Adam(optim_params, lr=opt['lr_G'], weight_decay=wd,
                                     betas=(beta1, beta2))
        return optimizer

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQs'].to(self.device)
        scale = data.get('scale', None)
        if isinstance(scale, (float, int)) or scale is None:
            self.scale = scale
        else:
            self.scale = data['scale'][0].item()
        if need_GT:
            self.real_H = data['GT'].to(self.device)

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        if self.ret_valid:
            self.fake_H, valid = self.netG(self.var_L, self.scale)
            if not valid:
                self.invalid_cnt += 1
                self.log_dict['l_pix'] = None
                self.log_dict['inval_cnt'] = self.invalid_cnt
                return None
        else:
            self.fake_H = self.netG(self.var_L, self.scale)

        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        l_pix.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()
        self.log_dict['inval_cnt'] = self.invalid_cnt

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            if self.opt['network_G']['which_model_G'] == 'EDVR3D':
                self.fake_H = self.netG(self.var_L.transpose(1,2), self.scale)
                self.fake_H = self.fake_H[:, :, 2, :, :]
            else:
                self.fake_H = self.netG(self.var_L, self.scale)
                if self.ret_valid:
                    self.fake_H = self.fake_H[0]
        self.netG.train()

class UPFlowVideoModel(UPVideoModel):
    def __init__(self, opt):
        super(UPFlowVideoModel, self).__init__(opt)
        self.netF = self.get_network(opt, 'F')
        self.nframe = self.opt['datasets']['train']['N_frames']
        self.half_nframe = self.nframe // 2
        self.l_flo_w = self.opt['train']['flow_weight']
        self.cri_flo = MaskedCharbonnierLoss(reduction=self.opt['train']['reduction'])
        self.load_F()

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQs'].to(self.device)
        # self.keys = data['key']
        scale = data.get('scale', None)
        if isinstance(scale, (float, int)) or scale is None:
            self.scale = scale
        else:
            self.scale = data['scale'][0].item()
        if need_GT:
            self.real_H = data['GT'].to(self.device)
        self.pre_H = data.get('Pre', None)
        self.if_pre = data.get('if_pre', (True, ))


        # self.pre_H = self.pre_H.to(self.device)
        # f_flow = Flow_arch.estimate_flow(self.netF, self.pre_H, self.real_H)
        # b_flow = Flow_arch.estimate_flow(self.netF, self.real_H, self.pre_H)
        # self.wrap_H, mask = Flow_arch.wraping(self.pre_H, b_flow)
        # self.valid_area = Flow_arch.detect_occlusion(f_flow, b_flow)
        # self.valid_area = self.valid_area.repeat(1,3,1,1)
        # plt.figure(0); idx=7
        # plt.subplot(241); 
        # plt.imshow((self.pre_H[idx].cpu().numpy().transpose((1,2,0))*255).astype(np.uint8))
        # plt.subplot(242); 
        # plt.imshow((self.real_H[idx].cpu().numpy().transpose((1,2,0))*255).astype(np.uint8))
        # plt.subplot(243); 
        # plt.imshow(f_flow[idx][0].cpu().numpy())
        # plt.subplot(244); 
        # plt.imshow(b_flow[idx][0].cpu().numpy())
        # plt.subplot(246); 
        # plt.imshow((self.wrap_H[idx].cpu().numpy().transpose((1,2,0))*255).astype(np.uint8))
        # plt.subplot(247); 
        # plt.imshow((mask[idx].cpu().numpy().transpose((1,2,0))*255).astype(np.uint8))
        # plt.subplot(248); 
        # plt.imshow((self.valid_area[idx].cpu().numpy().transpose((1,2,0))*255).astype(np.uint8))
        # plt.show()


    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()

        upX = F.interpolate(self.var_L[:, self.half_nframe, :, :, :], 
                            scale_factor=self.scale, mode='bilinear', align_corners=False)
        if not all(self.if_pre) or self.pre_H is None:
            self.wrap_H = upX
            self.valid_area = None
        else:
            self.pre_H = self.pre_H.to(self.device)
            upX1 = F.interpolate(self.var_L[:, self.half_nframe - 1, :, :, :], 
                            scale_factor=self.scale, mode='bilinear', align_corners=False)
            first = torch.cat([self.pre_H, self.real_H, upX])
            second = torch.cat([self.real_H, self.pre_H, upX1])
            flows = Flow_arch.estimate_flow(self.netF, first, second)
            f_flow, b_flow, t_flow = flows.chunk(3)
            self.wrap_H, _ = Flow_arch.wraping(self.pre_H, t_flow)
            self.valid_area = Flow_arch.detect_occlusion(f_flow, b_flow)

        if self.ret_valid:
            self.fake_H, valid = self.netG(self.var_L, self.wrap_H, self.scale)
            if not valid:
                self.invalid_cnt += 1
                self.log_dict['l_pix'] = None
                self.log_dict['inval_cnt'] = self.invalid_cnt
                return None
        else:
            self.fake_H = self.netG(self.var_L, self.wrap_H, self.scale)

        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        if self.valid_area is not None:
            l_flo = self.l_flo_w * self.cri_flo(self.fake_H, 
                        self.wrap_H, self.valid_area)
            loss = l_pix + l_flo
            self.log_dict['l_flo'] = l_flo.item()
        else:
            loss = l_pix
        loss.backward()
        self.optimizer_G.step()
        # set log
        self.log_dict['l_pix'] = l_pix.item()
        self.log_dict['loss'] = loss.item()
        self.log_dict['inval_cnt'] = self.invalid_cnt

    def test(self):
        self.netG.eval()
        upX = F.interpolate(self.var_L[:, self.half_nframe, :, :, :], 
                            scale_factor=self.scale, mode='bilinear', align_corners=False)
        if self.pre_H is None:
            self.wrap_H = upX
        else:
            first = F.interpolate(self.var_L[:, self.half_nframe - 1, :, :, :], 
                            scale_factor=self.scale, mode='bilinear', align_corners=False)
            second = upX
            b_flow = Flow_arch.estimate_flow(self.netF, second, first)
            self.pre_H = self.pre_H.unsqueeze(0).to(self.device)
            self.wrap_H, _ = Flow_arch.wraping(self.pre_H, b_flow)

        with torch.no_grad():
            self.fake_H = self.netG(self.var_L, self.wrap_H, self.scale)
            if self.ret_valid:
                self.fake_H = self.fake_H[0]
        self.netG.train()

    def load_F(self):
        load_path_F = self.opt['path']['pretrain_model_F']
        if load_path_F is not None:
            logger.info('Loading model for F [{:s}] ...'.format(load_path_F))
            self.load_network(load_path_F, self.netF, True)

class MetaVideoModel(VideoBaseModel):
    def __init__(self, opt):
        super(MetaVideoModel, self).__init__(opt)
        train_opt = self.opt['train']
        self.optimizer_G = self.get_optimizer(self.netG, train_opt)
        self.optimizers = [self.optimizer_G]

        self.all_scales = self.opt['scale']
        self.LQ_size = self.opt['datasets']['train']['LQ_size']
        self.pre_compute_W = self.opt['network_G']['pre_compute_W']
        self.Pos, self.Mask = None, None
        if self.pre_compute_W:
            self.Pos, self.Mask = {}, {}
            assert(self.LQ_size is not None), "LQ_size is {} for pre-compute-W ".format(self.LQ_size)
            if isinstance(self.all_scales, (list, tuple)):
                for scale in self.all_scales:
                    P, M = self.input_matrix_wpn(self.LQ_size, self.LQ_size, scale)
                    self.Pos[scale] = P
                    self.Mask[scale] = M
            elif isinstance(self.all_scales, (int, float)):
                P, M = self.input_matrix_wpn(self.LQ_size, self.LQ_size, self.all_scales)
                self.Pos[scale] = P
                self.Mask[scale] = M
            self.test_pros, self.test_masks = {}, {}

    def get_optimizer(self, net, opt):
        wd = opt['weight_decay_G'] if opt['weight_decay_G'] else 0
        beta1 = opt['beta1'] if opt['beta1'] else 0.9
        beta2 = opt['beta2'] if opt['beta2'] else 0.99

        if opt['edvr_lr_mult'] <= 0:
            optim_params = []
            for k, v in net.named_parameters():
                if v.requires_grad:
                    if 'P2W' in k:
                        optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
        else:
            edvr_params, meta_params = [], []
            for k, v in net.named_parameters():
                if 'P2W' in k:
                    meta_params.append(v)
                else:
                    edvr_params.append(v)
            optim_params = [{'params': edvr_params, 'name': 'edvr', 
                             'lr': opt['lr_G'] * opt['edvr_lr_mult']}, 
                            {'params': meta_params, 'name': 'meta', 
                            'lr': opt['lr_G']}]

        optimizer = torch.optim.Adam(optim_params, lr=opt['lr_G'],
                                    weight_decay=wd, betas=(beta1, beta2))
        return optimizer

    def input_matrix_wpn(self,inH, inW, scale, add_scale=True):
        '''
        inH, inW: the size of the feature maps
        scale: is the upsampling times
        '''
        outH, outW = int(scale*inH), int(scale*inW)

        #### mask records which pixel is invalid, 1 valid or 0 invalid
        #### h_offset and w_offset caculate the offset to generate the input matrix
        scale_int = int(math.ceil(scale))
        h_offset = torch.ones(inH, scale_int, 1)
        mask_h = torch.zeros(inH,  scale_int, 1)
        w_offset = torch.ones(1, inW, scale_int)
        mask_w = torch.zeros(1, inW, scale_int)
        if add_scale:
            scale_mat = torch.zeros(1,1)
            scale_mat[0,0] = 1.0/scale
            #res_scale = scale_int - scale
            #scale_mat[0,scale_int-1]=1-res_scale
            #scale_mat[0,scale_int-2]= res_scale
            scale_mat = torch.cat([scale_mat]*(inH*inW*(scale_int**2)),0)  
            ###(inH*inW*scale_int**2, 4)

        ####projection  coordinate  and caculate the offset 
        h_project_coord = torch.arange(0,outH, 1).float().mul(1.0/scale)
        int_h_project_coord = torch.floor(h_project_coord)

        offset_h_coord = h_project_coord - int_h_project_coord
        int_h_project_coord = int_h_project_coord.int()

        w_project_coord = torch.arange(0, outW, 1).float().mul(1.0/scale)
        int_w_project_coord = torch.floor(w_project_coord)

        offset_w_coord = w_project_coord - int_w_project_coord
        int_w_project_coord = int_w_project_coord.int()

        ####flag for   number for current coordinate LR image
        flag = 0
        number = 0
        for i in range(outH):
            if int_h_project_coord[i] == number:
                h_offset[int_h_project_coord[i], flag, 0] = offset_h_coord[i]
                mask_h[int_h_project_coord[i], flag,  0] = 1
                flag += 1
            else:
                h_offset[int_h_project_coord[i], 0, 0] = offset_h_coord[i]
                mask_h[int_h_project_coord[i], 0, 0] = 1
                number += 1
                flag = 1

        flag = 0
        number = 0
        for i in range(outW):
            if int_w_project_coord[i] == number:
                w_offset[0, int_w_project_coord[i], flag] = offset_w_coord[i]
                mask_w[0, int_w_project_coord[i], flag] = 1
                flag += 1
            else:
                w_offset[0, int_w_project_coord[i], 0] = offset_w_coord[i]
                mask_w[0, int_w_project_coord[i], 0] = 1
                number += 1
                flag = 1

        ## the size is scale_int* inH* (scal_int*inW)
        h_offset_coord = torch.cat([h_offset] * (scale_int * inW), 2).view(-1, scale_int * inW, 1)
        w_offset_coord = torch.cat([w_offset] * (scale_int * inH), 0).view(-1, scale_int * inW, 1)
        ####
        mask_h = torch.cat([mask_h] * (scale_int * inW), 2).view(-1, scale_int * inW, 1)
        mask_w = torch.cat([mask_w] * (scale_int * inH), 0).view(-1, scale_int * inW, 1)

        pos_mat = torch.cat((h_offset_coord, w_offset_coord), 2)
        mask_mat = torch.sum(torch.cat((mask_h,mask_w),2),2).view(scale_int*inH,scale_int*inW)
        mask_mat = mask_mat.eq(2)
        pos_mat = pos_mat.contiguous().view(1, -1,2)
        if add_scale:
            pos_mat = torch.cat((scale_mat.view(1,-1,1), pos_mat),2)

        return pos_mat.to(self.device), mask_mat.to(self.device)
               # #outH*outW*2 outH=scale_int*inH, outW=scale_int*inW

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQs'].to(self.device)
        scale = data.get('scale', None)
        if isinstance(scale, (float, int)) or scale is None:
            self.scale = scale
        else:
            self.scale = data['scale'][0].item()
        if need_GT:
            self.real_H = data['GT'].to(self.device)

    def optimize_parameters(self, step):
        if self.opt['train']['ft_tsa_only'] and step < self.opt['train']['ft_tsa_only']:
            self.set_params_lr_zero()

        N, K, C, H, W = self.var_L.size()
        _, _, outH, outW = self.real_H.size()

        if self.pre_compute_W:
            scale_coord_map = self.Pos[self.scale].clone()
            mask = self.Mask[self.scale].clone()
        else:
            scale_coord_map, mask = self.input_matrix_wpn(H, W, self.scale)
        if self.n_gpus > 1 and not self.opt['dist']:
            scale_coord_map = torch.cat([scale_coord_map] * self.n_gpus, 0)
            mask = torch.cat([mask] * self.n_gpus, 0)
        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L, self.scale, scale_coord_map, mask)

        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        l_pix.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            N, K, C, H, W = self.var_L.size()
            if self.pre_compute_W and (H, W, self.scale) not in self.test_masks:
                scale_coord_map, mask = self.input_matrix_wpn(H, W, self.scale)
                self.test_pros[(H, W, self.scale)] = scale_coord_map.clone()
                self.test_masks[(H, W, self.scale)] = mask.clone()
            else:
                scale_coord_map = self.test_pros[(H, W, self.scale)].clone()
                mask = self.test_masks[(H, W, self.scale)].clone()
            if self.n_gpus > 1 and not self.opt['dist']:
                scale_coord_map = torch.cat([scale_coord_map] * self.n_gpus, 0)
                mask = torch.cat([mask] * self.n_gpus, 0)
            self.fake_H = self.netG(self.var_L, self.scale, scale_coord_map, mask)
        self.netG.train()
