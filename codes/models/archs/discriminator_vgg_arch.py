import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.utils.spectral_norm as spectral_norm

from modules.deform_conv3d import (DeformConv3d, _DeformConv3d, 
                                DeformConv3dPack, DeformConv3dPack_v2)

class Discriminator_VGG_128(nn.Module):
    def __init__(self, in_nc, nf):
        super(Discriminator_VGG_128, self).__init__()
        # [64, 128, 128]
        self.conv0_0 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(nf, affine=True)
        # [64, 64, 64]
        self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(nf * 2, affine=True)
        self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(nf * 2, affine=True)
        # [128, 32, 32]
        self.conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(nf * 4, affine=True)
        self.conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(nf * 4, affine=True)
        # [256, 16, 16]
        self.conv3_0 = nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv3_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(nf * 8, affine=True)
        # [512, 8, 8]
        self.conv4_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv4_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(nf * 8, affine=True)

        self.linear1 = nn.Linear(512 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.bn0_1(self.conv0_1(fea)))

        fea = self.lrelu(self.bn1_0(self.conv1_0(fea)))
        fea = self.lrelu(self.bn1_1(self.conv1_1(fea)))

        fea = self.lrelu(self.bn2_0(self.conv2_0(fea)))
        fea = self.lrelu(self.bn2_1(self.conv2_1(fea)))

        fea = self.lrelu(self.bn3_0(self.conv3_0(fea)))
        fea = self.lrelu(self.bn3_1(self.conv3_1(fea)))

        fea = self.lrelu(self.bn4_0(self.conv4_0(fea)))
        fea = self.lrelu(self.bn4_1(self.conv4_1(fea)))

        fea = fea.view(fea.size(0), -1)
        fea = self.lrelu(self.linear1(fea))
        out = self.linear2(fea)
        return out

class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=34, use_bn=False, use_input_norm=True,
                 device=torch.device('cpu')):
        super(VGGFeatureExtractor, self).__init__()
        self.use_input_norm = use_input_norm
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485 - 1, 0.456 - 1, 0.406 - 1] if input in range [-1, 1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229 * 2, 0.224 * 2, 0.225 * 2] if input in range [-1, 1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        # Assume input range is [0, 1]
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


def weights_init(m):
    classname = m.__class__.__name__
    no_param_list = ["InterNet", "DeconvUpBlock", "MultiscaleDiscriminator", \
                     "AvgPool2d", "Sequential", "LeakyReLU", "ReplicationPad3d", \
                     "ReLU", "Res3DBlock", "Tanh", "ResNet", "AvgPool3d", \
                     "AvgPool2d", "MultiscaleDiscriminator_2D", "FeatExpand", \
                     "ConstantPad3d", "PredNet"]
    if classname.find('Conv3d') != -1 and classname.find('ShiftConv3d') == -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('ConvTranspose3d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm3d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('ConvTranspose2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    else:
        if classname not in no_param_list:
            print("in weights_init ")
            print(classname)
    if classname == "DeformConv3dPack" or classname == "DeformConv3dPack_v2":
        m.conv_offset_mask.weight.data.zero_()
        m.conv_offset_mask.bias.data.zero_()
    # print('init: ' + classname)

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf, n_layers=3, norm_layer=nn.BatchNorm3d, 
        use_sigmoid=False, num_D=1, get_inter_feat=False, has_bias=False, 
        has_sn=True, max_ndf=256, conv_type='deform'):
        super(MultiscaleDiscriminator, self).__init__()
        self.input_nc = input_nc
        self.ndf = ndf
        self.n_layers = n_layers
        self.norm_layer = norm_layer
        self.use_sigmoid = use_sigmoid
        self.num_D = num_D
        self.get_inter_feat = get_inter_feat
        self.has_bias = has_bias
        self.has_sn = has_sn
        self.max_ndf = max_ndf
        self.conv_type = conv_type

        for i in range(self.num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, 
                use_sigmoid, get_inter_feat, has_bias, has_sn, max_ndf, conv_type)
            if self.get_inter_feat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool3d(kernel_size=[1, 3, 3], stride=[1, 2, 2], 
            padding=[0, 1, 1], count_include_pad=False)

        self.apply(weights_init)

    def singleD_forward(self, model, input):
        if self.get_inter_feat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[-2:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.get_inter_feat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i)+'_layer' + str(j)) \
                            for j in range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer'+str(num_D - 1 - i))
            
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, 
        use_sigmoid=False, get_inter_feat=False, has_bias=False, has_sn=False, 
        max_ndf=256, conv_type="deform"):
        super(NLayerDiscriminator, self).__init__()
        self.input_nc = input_nc
        self.ndf = ndf
        self.n_layers = n_layers
        self.norm_layer = norm_layer
        self.use_sigmoid = use_sigmoid
        self.get_inter_feat = get_inter_feat
        self.has_bias = has_bias
        self.has_sn = has_sn
        self.max_ndf = max_ndf
        self.conv_type = conv_type

        conv1 = nn.Conv3d(input_nc, ndf, kernel_size=(3, 3, 3), stride=(1, 2, 2), 
                            padding=(0, 0, 0), bias=has_bias)
        if has_sn == True:
            conv1 = spectral_norm(conv1)

        pad1 = nn.ReplicationPad3d((1, 1, 1, 1, 1, 1))
        sequence = [[pad1, conv1, nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(0, n_layers):
            nf_prev = nf
            nf = min(nf * 2, max_ndf)
            if conv_type == "deform":
                conv_s = DeformConv3dPack(nf_prev, nf, kernel_size=(3, 3, 3), 
                            stride=(1, 2, 2), padding=(0, 0, 0), bias=has_bias)
            elif conv_type == "deformv2":
                conv_s = DeformConv3dPack_v2(nf_prev, nf, kernel_size=(3, 3, 3), 
                            stride=(1, 2, 2),padding=(0, 0, 0),bias=has_bias)
            elif conv_type == "normal":
                conv_s = nn.Conv3d(nf_prev, nf, kernel_size=(3,3,3), 
                                stride=(1, 2, 2), padding=(0, 0, 0), bias=has_bias)
            else:
                print("NotImplementedError for conv in D ")

            if has_sn == True:
                conv_s = spectral_norm(conv_s)

            sequence += [[pad1, conv_s, norm_layer(nf), nn.LeakyReLU(0.2, True)]]

        nf_prev = nf
        nf = min(nf * 2, max_ndf)

        pad2 = nn.ReplicationPad3d((1, 1, 1, 1, 0, 0))
        conv3 = nn.Conv3d(nf,1,kernel_size=(1, 3, 3),stride=(1, 1, 1), 
            padding=(0, 0, 0), bias=has_bias)

        if has_sn == True:
            conv3 = spectral_norm(conv3)

        sequence += [[pad2, conv3]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if get_inter_feat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.get_inter_feat:
            res = [input]
            for n in range(self.n_layers + 2):
                # print("debug in netD --- ")
                # print(n)
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[-2:]
        else:
            return self.model(input)
