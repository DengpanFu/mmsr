import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']
    # image restoration
    if model == 'sr':  # PSNR-oriented super resolution
        from .SR_model import SRModel as M
    elif model == 'srgan':  # GAN-based super resolution, SRGAN / ESRGAN
        from .SRGAN_model import SRGANModel as M
    # video restoration
    elif model == 'video_base':
        from .Video_base_model import VideoBaseModel as M
    elif model == 'meta_video':
        from .Video_base_model import MetaVideoModel as M
    elif model == 'up_video':
        from .Video_base_model import UPVideoModel as M
    elif model == 'up_flow_video':
        from .Video_base_model import UPFlowVideoModel as M
    elif model == 'video_3d_D':
        from .Video_3dD_model import Videl3dDModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
