import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from modules.deform_conv3d import DeformConv3d, DeformConv3dPack, DeformConv3dPack_v2

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

class ResidualBlock_IN(nn.Module):
    '''Residual block InstanceNorm
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_IN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.norm1 = nn.InstanceNorm2d(nf, affine=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.norm2 = nn.InstanceNorm2d(nf, affine=True)
        self.relu = nn.ReLU(True)

        # initialization
        # initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.relu(self.norm2(self.conv2(out)))
        return identity + out

class ResidualBlock(nn.Module):
    '''Residual block InstanceNorm
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64, norm=None):
        super(ResidualBlock_IN, self).__init__()
        self.nf = nf
        self.norm = norm
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if not norm is None:
            if norm.lower() == 'in':
                self.norm1 = nn.InstanceNorm2d(nf, affine=True)
                self.norm2 = nn.InstanceNorm2d(nf, affine=True)
            elif norm.lower == 'bn':
                self.norm1 = nn.BatchNorm2d(nf)
                self.norm2 = nn.BatchNorm2d(nf)
        self.relu = nn.ReLU(True)

        init_modules = [self.conv1, self.conv2]
        # initialization
        if not norm is None:
            init_modules.extend([self.norm1, self.norm2])
        initialize_weights(init_modules, 0.1)

    def forward(self, x):
        identity = x
        if not self.norm is None:
            out = self.relu(self.norm1(self.conv1(x)))
            out = self.relu(self.norm2(self.conv2(out)))
        else:
            out = self.relu(self.conv1(x))
            out = self.conv2(out)
        return identity + out

class ResidualBlock_3D(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, planes=64, conv_type='normal', padding_mode='const', 
        has_bias=True, if_relu=False):
        super(ResidualBlock_3D, self).__init__()
        self.planes = planes
        self.conv_type = conv_type
        self.padding_mode = padding_mode
        self.has_bias = has_bias
        self.if_relu = if_relu

        if self.conv_type == "deformv2":
            self.conv1 = DeformConv3dPack_v2(planes, planes, kernel_size=(3, 3, 3), 
                            stride=(1, 1, 1), padding=(0, 0, 0), bias=has_bias)
        elif self.conv_type == "deform":
            self.conv1 = DeformConv3dPack(planes, planes, kernel_size=(3, 3, 3), 
                            stride=(1, 1, 1), padding=(0, 0, 0), bias=has_bias)
        elif self.conv_type == "normal":
            self.conv1 = nn.Conv3d(planes, planes, kernel_size=(3, 3, 3), 
                            stride=(1, 1, 1), padding=(0, 0, 0), bias=has_bias)
        else:
            raise TypeError('Unknown Conv3d type: {}'.format(self.conv_type))

        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(3, 3, 3), 
                        stride=(1, 1, 1), padding=(0, 0, 0), bias=has_bias)
        self.bn2 = nn.BatchNorm3d(planes)

        if padding_mode == "const":
            self.pad1 = nn.ReplicationPad3d((1,1,1,1,1,1))
            self.pad2 = nn.ReplicationPad3d((1,1,1,1,1,1))
        elif padding_mode == "zero":
            self.pad1 = nn.ConstantPad3d((1,1,1,1,1,1),0)
            self.pad2 = nn.ConstantPad3d((1,1,1,1,1,1),0)
        else:
            raise TypeError("Wrong padding: {}, Only support padding mode: (const, zero)".
                                format(self.padding_mode))

        self.model = nn.Sequential(self.pad1, self.conv1, self.bn1, 
                            self.relu, self.pad2, self.conv2, self.bn2)

        # initialization
        self.init_weights()

    def forward(self, x):
        out = x + self.model(x)
        if self.if_relu:
            out = self.relu(out)
        return out

    def init_weights(self, scale=0.1):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, DeformConv3dPack, DeformConv3dPack_v2)):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output
