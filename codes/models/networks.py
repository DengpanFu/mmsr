import torch
import models.archs.SRResNet_arch as SRResNet_arch
import models.archs.discriminator_vgg_arch as SRGAN_arch
import models.archs.RRDBNet_arch as RRDBNet_arch
import models.archs.EDVR_arch as EDVR_arch
# import models.archs.Flow_arch as Flow_arch

# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    # image restoration
    if which_model == 'MSRResNet':
        netG = SRResNet_arch.MSRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'RRDBNet':
        netG = RRDBNet_arch.RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'])
    # video restoration
    elif which_model == 'EDVR':
        netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                              predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                              w_TSA=opt_net['w_TSA'])
    elif which_model == 'EDVR2X':
        netG = EDVR_arch.EDVR2X(nf=opt_net['nf'], nframes=opt_net['nframes'],
                                groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                                back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                                predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                                w_TSA=opt_net['w_TSA'])
    elif which_model == 'EDVRImg':
        netG = EDVR_arch.EDVRImage(nf=opt_net['nf'], front_RBs=opt_net['front_RBs'],
                                 back_RBs=opt_net['back_RBs'], down_scale=opt_net['down_scale'])
    elif which_model == 'EDVR3D':
        netG = EDVR_arch.EDVR3D(nf=opt_net['nf'], front_RBs=opt_net['front_RBs'],
                                back_RBs=opt_net['back_RBs'], down_scale=opt_net['down_scale'])
    elif which_model == 'UPEDVR':
        netG = EDVR_arch.UPEDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                                groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                                back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                                w_TSA=opt_net['w_TSA'], down_scale=opt_net['down_scale'], 
                                align_target=opt_net['align_target'], 
                                ret_valid=opt_net['ret_valid'])
    elif which_model == 'UPContEDVR':
        netG = EDVR_arch.UPControlEDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                                       groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                                       back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                                       w_TSA=opt_net['w_TSA'], down_scale=opt_net['down_scale'], 
                                       align_target=opt_net['align_target'], 
                                       ret_valid=opt_net['ret_valid'], 
                                       multi_scale_cont=opt_net['multi_scale_cont'])
    elif which_model == 'FlowUPContEDVR':
        netG = EDVR_arch.FlowUPControlEDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                                           groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                                           back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                                           w_TSA=opt_net['w_TSA'], down_scale=opt_net['down_scale'], 
                                           align_target=opt_net['align_target'], 
                                           ret_valid=opt_net['ret_valid'], 
                                           multi_scale_cont=opt_net['multi_scale_cont'])
    # video SR for multiple target frames
    elif which_model == 'MultiEDVR':
        netG = EDVR_arch.MultiEDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                                   groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                                   back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                                   predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                                   w_TSA=opt_net['w_TSA'])
    # arbitrary magnification video super-resolution
    elif which_model == 'MetaEDVR':
        netG = EDVR_arch.MetaEDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                                  groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                                  back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                                  predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                                  w_TSA=opt_net['w_TSA'], fix_edvr=opt_net['fix_edvr'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG


# Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    elif which_model == 'DCN3D':
        netD = SRGAN_arch.MultiscaleDiscriminator(input_nc=opt_net['input_nc'], 
                                                  ndf=opt_net['ndf'], 
                                                  n_layers=opt_net['n_layers'], 
                                                  num_D=opt_net['num_d'], 
                                                  use_sigmoid=opt_net['use_sigmoid'], 
                                                  get_inter_feat=opt_net['get_inter_feat'], 
                                                  has_bias=opt_net['has_bias'], 
                                                  has_sn=opt_net['has_sn'], 
                                                  max_ndf=opt_net['max_ndf'], 
                                                  conv_type=opt_net['conv_type'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD


# Define network used for perceptual loss
def define_P(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = SRGAN_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF

# Define network for Optical Flow
def define_F(opt):
    netF = Flow_arch.PWCNet()
    netF.eval()
    return netF