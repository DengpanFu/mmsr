''' network architecture for EDVR '''
import functools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util

from models.archs.dcn.deform_conv import ModulatedDeformConvPack as DCN

class Predeblur_ResNet_Pyramid(nn.Module):
    def __init__(self, nf=128, HR_in=False):
        '''
        HR_in: True if the inputs are high spatial size
        '''

        super(Predeblur_ResNet_Pyramid, self).__init__()
        self.HR_in = True if HR_in else False
        if self.HR_in:
            self.conv_first_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
            self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
            self.conv_first_3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        else:
            self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        self.RB_L1_1 = basic_block()
        self.RB_L1_2 = basic_block()
        self.RB_L1_3 = basic_block()
        self.RB_L1_4 = basic_block()
        self.RB_L1_5 = basic_block()
        self.RB_L2_1 = basic_block()
        self.RB_L2_2 = basic_block()
        self.RB_L3_1 = basic_block()
        self.deblur_L2_conv = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.deblur_L3_conv = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        if self.HR_in:
            L1_fea = self.lrelu(self.conv_first_1(x))
            L1_fea = self.lrelu(self.conv_first_2(L1_fea))
            L1_fea = self.lrelu(self.conv_first_3(L1_fea))
        else:
            L1_fea = self.lrelu(self.conv_first(x))
        L2_fea = self.lrelu(self.deblur_L2_conv(L1_fea))
        L3_fea = self.lrelu(self.deblur_L3_conv(L2_fea))
        L3_fea = F.interpolate(self.RB_L3_1(L3_fea), scale_factor=2, mode='bilinear',
                               align_corners=False)
        L2_fea = self.RB_L2_1(L2_fea) + L3_fea
        L2_fea = F.interpolate(self.RB_L2_2(L2_fea), scale_factor=2, mode='bilinear',
                               align_corners=False)
        L1_fea = self.RB_L1_2(self.RB_L1_1(L1_fea)) + L2_fea
        out = self.RB_L1_5(self.RB_L1_4(self.RB_L1_3(L1_fea)))
        return out

class PCD_Align(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    '''

    def __init__(self, nf=64, groups=8):
        super(PCD_Align, self).__init__()
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        self.L2_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        self.L1_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # Cascading DCN
        self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.cas_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                               extra_offset_mask=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_fea_l, ref_fea_l):
        '''align other neighboring frames to the reference frame in the feature level
        nbr_fea_l, ref_fea_l: [L1, L2, L3], each with [B,C,H,W] features
        '''
        # L3
        L3_offset = torch.cat([nbr_fea_l[2], ref_fea_l[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack([nbr_fea_l[2], L3_offset]))
        # L2
        L2_offset = torch.cat([nbr_fea_l[1], ref_fea_l[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
        L2_fea = self.L2_dcnpack([nbr_fea_l[1], L2_offset])
        L3_fea = F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        L1_offset = torch.cat([nbr_fea_l[0], ref_fea_l[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
        L1_fea = self.L1_dcnpack([nbr_fea_l[0], L1_offset])
        L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv(torch.cat([L1_fea, L2_fea], dim=1))
        # Cascading
        offset = torch.cat([L1_fea, ref_fea_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea = self.lrelu(self.cas_dcnpack([L1_fea, offset]))

        return L1_fea

class PCD_Align_Valid(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    '''

    def __init__(self, nf=64, groups=8):
        super(PCD_Align_Valid, self).__init__()
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True, return_valid=True)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True, return_valid=True)
        self.L2_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True, return_valid=True)
        self.L1_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # Cascading DCN
        self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.cas_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                               extra_offset_mask=True, return_valid=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_fea_l, ref_fea_l):
        '''align other neighboring frames to the reference frame in the feature level
        nbr_fea_l, ref_fea_l: [L1, L2, L3], each with [B,C,H,W] features
        '''
        # L3
        L3_offset = torch.cat([nbr_fea_l[2], ref_fea_l[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
        L3_offset, valid3 = self.L3_dcnpack([nbr_fea_l[2], L3_offset])
        L3_fea = self.lrelu(L3_offset)
        # L2
        L2_offset = torch.cat([nbr_fea_l[1], ref_fea_l[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
        L2_fea, valid2 = self.L2_dcnpack([nbr_fea_l[1], L2_offset])
        L3_fea = F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        L1_offset = torch.cat([nbr_fea_l[0], ref_fea_l[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
        L1_fea, valid1 = self.L1_dcnpack([nbr_fea_l[0], L1_offset])
        L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv(torch.cat([L1_fea, L2_fea], dim=1))
        # Cascading
        offset = torch.cat([L1_fea, ref_fea_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea, valid0 = self.cas_dcnpack([L1_fea, offset])
        L1_fea = self.lrelu(L1_fea)
        valid = valid3 & valid2 & valid1 & valid0
        return L1_fea, valid

class TSA_Fusion(nn.Module):
    ''' Temporal Spatial Attention fusion module
    Temporal: correlation;
    Spatial: 3 pyramid levels.
    '''

    def __init__(self, nf=64, nframes=5, center=2):
        super(TSA_Fusion, self).__init__()
        self.center = center
        # temporal attention (before fusion conv)
        self.tAtt_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.tAtt_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # fusion conv: using 1x1 to save parameters and computation
        self.fea_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)

        # spatial attention (after fusion conv)
        self.sAtt_1 = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(3, stride=2, padding=1)
        self.sAtt_2 = nn.Conv2d(nf * 2, nf, 1, 1, bias=True)
        self.sAtt_3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_4 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_5 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_L1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_L2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.sAtt_L3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_add_1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_add_2 = nn.Conv2d(nf, nf, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, aligned_fea):
        B, N, C, H, W = aligned_fea.size()  # N video frames
        #### temporal attention
        emb_ref = self.tAtt_2(aligned_fea[:, self.center, :, :, :].clone())
        emb = self.tAtt_1(aligned_fea.view(-1, C, H, W)).view(B, N, -1, H, W)  # [B, N, C(nf), H, W]

        cor_l = []
        for i in range(N):
            emb_nbr = emb[:, i, :, :, :]
            cor_tmp = torch.sum(emb_nbr * emb_ref, 1).unsqueeze(1)  # B, 1, H, W
            cor_l.append(cor_tmp)
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))  # B, N, H, W
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1).view(B, -1, H, W)
        aligned_fea = aligned_fea.view(B, -1, H, W) * cor_prob

        #### fusion
        fea = self.lrelu(self.fea_fusion(aligned_fea))

        #### spatial attention
        att = self.lrelu(self.sAtt_1(aligned_fea))
        att_max = self.maxpool(att)
        att_avg = self.avgpool(att)
        att = self.lrelu(self.sAtt_2(torch.cat([att_max, att_avg], dim=1)))
        # pyramid levels
        att_L = self.lrelu(self.sAtt_L1(att))
        att_max = self.maxpool(att_L)
        att_avg = self.avgpool(att_L)
        att_L = self.lrelu(self.sAtt_L2(torch.cat([att_max, att_avg], dim=1)))
        att_L = self.lrelu(self.sAtt_L3(att_L))
        att_L = F.interpolate(att_L, scale_factor=2, mode='bilinear', align_corners=False)

        att = self.lrelu(self.sAtt_3(att))
        att = att + att_L
        att = self.lrelu(self.sAtt_4(att))
        att = F.interpolate(att, scale_factor=2, mode='bilinear', align_corners=False)
        att = self.sAtt_5(att)
        att_add = self.sAtt_add_2(self.lrelu(self.sAtt_add_1(att)))
        att = torch.sigmoid(att)

        fea = fea * att * 2 + att_add
        return fea

class EDVR(nn.Module):
    def __init__(self, nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10, center=None,
                 predeblur=False, HR_in=False, w_TSA=True):
        super(EDVR, self).__init__()
        self.nf = nf
        self.center = nframes // 2 if center is None else center
        self.is_predeblur = True if predeblur else False
        self.HR_in = True if HR_in else False
        self.w_TSA = w_TSA
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)

        #### extract features (for each frame)
        if self.is_predeblur:
            self.pre_deblur = Predeblur_ResNet_Pyramid(nf=nf, HR_in=self.HR_in)
            self.conv_1x1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        else:
            if self.HR_in:
                self.conv_first_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
                self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
                self.conv_first_3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
            else:
                self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.pcd_align = PCD_Align(nf=nf, groups=groups)
        if self.w_TSA:
            self.tsa_fusion = TSA_Fusion(nf=nf, nframes=nframes, center=self.center)
        else:
            self.tsa_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)

        #### reconstruction
        self.recon_trunk = arch_util.make_layer(ResidualBlock_noBN_f, back_RBs)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        B, N, C, H, W = x.size()  # N video frames
        x_center = x[:, self.center, :, :, :].contiguous()

        #### extract LR features
        # L1
        if self.is_predeblur:
            L1_fea = self.pre_deblur(x.view(-1, C, H, W))
            L1_fea = self.conv_1x1(L1_fea)
            if self.HR_in:
                H, W = H // 4, W // 4
        else:
            if self.HR_in:
                L1_fea = self.lrelu(self.conv_first_1(x.view(-1, C, H, W)))
                L1_fea = self.lrelu(self.conv_first_2(L1_fea))
                L1_fea = self.lrelu(self.conv_first_3(L1_fea))
                H, W = H // 4, W // 4
            else:
                L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))  # BN,64,64,64
        L1_fea = self.feature_extraction(L1_fea)   # BN,64,64,64
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))  # BN,64,32,32
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))  # BN,64,32,32
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))  # BN,64,16,16
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))  # BN,64,16,16

        L1_fea = L1_fea.view(B, N, -1, H, W)            # B,N,64,64,64
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)  # B,N,64,32,32
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)  # B,N,64,16,16

        #### pcd align
        # ref feature list
        ref_fea_l = [
            L1_fea[:, self.center, :, :, :].clone(), 
            L2_fea[:, self.center, :, :, :].clone(),
            L3_fea[:, self.center, :, :, :].clone()
        ]
        aligned_fea = []
        for i in range(N):
            nbr_fea_l = [
                L1_fea[:, i, :, :, :].clone(), 
                L2_fea[:, i, :, :, :].clone(),
                L3_fea[:, i, :, :, :].clone()
            ]
            aligned_fea.append(self.pcd_align(nbr_fea_l, ref_fea_l))
        aligned_fea = torch.stack(aligned_fea, dim=1)  # [B, N, C, H, W]

        if not self.w_TSA:
            aligned_fea = aligned_fea.view(B, -1, H, W)
        fea = self.tsa_fusion(aligned_fea)

        out = self.recon_trunk(fea)   # B,C,64,64
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))  # B,C,128,128
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))  # B,C,256,256
        out = self.lrelu(self.HRconv(out))    # B,C,256,256
        out = self.conv_last(out)    # B,3,256,256
        if self.HR_in:
            base = x_center
        else:
            base = F.interpolate(x_center, scale_factor=4, mode='bilinear', 
                        align_corners=False)    # B,3,256,256
        out += base
        return out

class EDVR2X(nn.Module):
    def __init__(self, nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10, center=None,
                 predeblur=False, HR_in=False, w_TSA=True):
        super(EDVR2X, self).__init__()
        self.nf = nf
        self.center = nframes // 2 if center is None else center
        self.is_predeblur = True if predeblur else False
        self.HR_in = True if HR_in else False
        self.w_TSA = w_TSA
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)

        #### extract features (for each frame)
        if self.is_predeblur:
            self.pre_deblur = Predeblur_ResNet_Pyramid(nf=nf, HR_in=self.HR_in)
            self.conv_1x1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        else:
            if self.HR_in:
                self.conv_first_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
                self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
                self.conv_first_3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
            else:
                self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.pcd_align = PCD_Align(nf=nf, groups=groups)
        if self.w_TSA:
            self.tsa_fusion = TSA_Fusion(nf=nf, nframes=nframes, center=self.center)
        else:
            self.tsa_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)

        #### reconstruction
        self.recon_trunk = arch_util.make_layer(ResidualBlock_noBN_f, back_RBs)
        #### upsampling
        self.upconv = nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        B, N, C, H, W = x.size()  # N video frames
        x_center = x[:, self.center, :, :, :].contiguous()

        #### extract LR features
        # L1
        if self.is_predeblur:
            L1_fea = self.pre_deblur(x.view(-1, C, H, W))
            L1_fea = self.conv_1x1(L1_fea)
            if self.HR_in:
                H, W = H // 4, W // 4
        else:
            if self.HR_in:
                L1_fea = self.lrelu(self.conv_first_1(x.view(-1, C, H, W)))
                L1_fea = self.lrelu(self.conv_first_2(L1_fea))
                L1_fea = self.lrelu(self.conv_first_3(L1_fea))
                H, W = H // 4, W // 4
            else:
                L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))  # BN,64,64,64
        L1_fea = self.feature_extraction(L1_fea)   # BN,64,64,64
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))  # BN,64,32,32
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))  # BN,64,32,32
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))  # BN,64,16,16
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))  # BN,64,16,16

        L1_fea = L1_fea.view(B, N, -1, H, W)            # B,N,64,64,64
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)  # B,N,64,32,32
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)  # B,N,64,16,16

        #### pcd align
        # ref feature list
        ref_fea_l = [
            L1_fea[:, self.center, :, :, :].clone(), 
            L2_fea[:, self.center, :, :, :].clone(),
            L3_fea[:, self.center, :, :, :].clone()
        ]
        aligned_fea = []
        for i in range(N):
            nbr_fea_l = [
                L1_fea[:, i, :, :, :].clone(), 
                L2_fea[:, i, :, :, :].clone(),
                L3_fea[:, i, :, :, :].clone()
            ]
            aligned_fea.append(self.pcd_align(nbr_fea_l, ref_fea_l))
        aligned_fea = torch.stack(aligned_fea, dim=1)  # [B, N, C, H, W]

        if not self.w_TSA:
            aligned_fea = aligned_fea.view(B, -1, H, W)
        fea = self.tsa_fusion(aligned_fea)

        out = self.recon_trunk(fea)   # B,C,64,64
        out = self.lrelu(self.pixel_shuffle(self.upconv(out)))  # B,C,128,128
        out = self.lrelu(self.HRconv(out))    # B,C,128,128
        out = self.conv_last(out)    # B,3,128,128
        if self.HR_in:
            base = x_center
        else:
            base = F.interpolate(x_center, scale_factor=2, mode='bilinear', 
                        align_corners=False)    # B,3,128,128
        out += base
        return out

class EDVRImage(nn.Module):
    def __init__(self, nf=64, front_RBs=5, back_RBs=10, down_scale=True):
        super(EDVRImage, self).__init__()
        self.nf = nf
        self.front_RBs = front_RBs
        self.back_RBs = back_RBs
        self.down_scale = down_scale
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)

        #### extract features (for each frame)
        if self.down_scale:
            self.conv_first_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
            self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
            self.conv_first_3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        else:
            self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)

        #### reconstruction
        self.recon_trunk = arch_util.make_layer(ResidualBlock_noBN_f, back_RBs)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, scale):
        N, C, H, W = x.size()
        if self.down_scale:
            x = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
            base = x
            feat = self.lrelu(self.conv_first_1(x))
            feat = self.lrelu(self.conv_first_2(feat))
            feat = self.lrelu(self.conv_first_3(feat))
            # H, W = H // 4, W // 4
        else:
            base = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
            feat = self.lrelu(self.conv_first(x))
        feat = self.feature_extraction(feat)   # BN,64,64,64

        out = self.recon_trunk(feat)   # B,C,64,64
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))  # B,C,128,128
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))  # B,C,256,256
        out = self.lrelu(self.HRconv(out))    # B,C,256,256
        out = self.conv_last(out)    # B,3,256,256
        out += base
        return out

class EDVR3D(nn.Module):
    def __init__(self, nf=64, front_RBs=5, back_RBs=10, conv_type='normal', 
        down_scale=True):
        super(EDVR3D, self).__init__()
        self.nf = nf
        self.front_RBs = front_RBs
        self.back_RBs = back_RBs
        self.conv_type = conv_type
        self.down_scale = down_scale
        ResidualBlock_3D = functools.partial(arch_util.ResidualBlock_3D, 
            planes=nf, conv_type=conv_type, padding_mode='const', has_bias=True)

        #### extract features (for each frame)
        if self.down_scale:
            self.conv_first_1 = nn.Conv3d(3, nf, kernel_size=(3, 3, 3), 
                        stride=(1, 1, 1), padding=(1, 1, 1), bias=True)
            self.conv_first_2 = nn.Conv3d(nf, nf, kernel_size=(3, 3, 3), 
                        stride=(1, 2, 2), padding=(1, 1, 1), bias=True)
            self.conv_first_3 = nn.Conv3d(nf, nf, kernel_size=(3, 3, 3), 
                        stride=(1, 2, 2), padding=(1, 1, 1), bias=True)
        else:
            self.conv_first = nn.Conv3d(3, nf, kernel_size=(3, 3, 3), 
                        stride=(1, 1, 1), padding=(1, 1, 1), bias=True)
        self.feature_extraction = arch_util.make_layer(ResidualBlock_3D, front_RBs)

        #### reconstruction
        self.recon_trunk = arch_util.make_layer(ResidualBlock_3D, back_RBs)
        #### upsampling
        self.upconv1 = nn.ConvTranspose3d(nf, nf, kernel_size=(3, 3, 3), 
            stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1), bias=True)
        self.upconv2 = nn.ConvTranspose3d(nf, 64, kernel_size=(3, 3, 3), 
            stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1), bias=True)

        self.HRconv = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), 
                        stride=(1, 1, 1), padding=(1, 1, 1), bias=True)
        self.conv_last = nn.Conv3d(64, 3, kernel_size=(3, 3, 3), 
                        stride=(1, 1, 1), padding=(1, 1, 1), bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, scale):
        N, C, T, H, W = x.size()
        x = x.view(N, C*T, H, W)
        x = F.interpolate(x, scale_factor=scale, mode='bilinear', 
                                 align_corners=False)
        H, W = x.shape[-2:]
        x = x.view(N, C, T, H, W)
        base = x.detach()
        if self.down_scale:
            feat = self.lrelu(self.conv_first_1(x))
            feat = self.lrelu(self.conv_first_2(feat))
            feat = self.lrelu(self.conv_first_3(feat))
            # H, W = H // 4, W // 4
        else:
            feat = self.lrelu(self.conv_first(x))
        feat = self.feature_extraction(feat)

        out = self.recon_trunk(feat)
        out = self.lrelu(self.upconv1(out))
        out = self.lrelu(self.upconv2(out))
        out = self.lrelu(self.HRconv(out))
        out = self.conv_last(out)
        out += base
        return out

class UPEDVR(nn.Module):
    def __init__(self, nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10, 
        center=None, w_TSA=True, down_scale=False, align_target=True, ret_valid=False):
        super(UPEDVR, self).__init__()
        self.nf = nf
        self.nframes = nframes
        self.center = nframes // 2 if center is None else center
        self.w_TSA = w_TSA
        self.down_scale = down_scale
        self.align_target = align_target
        self.ret_valid = ret_valid
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)

        #### extract features (for each frame)
        if self.down_scale:
            self.conv_first_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
            self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
            self.conv_first_3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        else:
            self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if self.ret_valid:
            self.pcd_align = PCD_Align_Valid(nf=nf, groups=groups)
        else:
            self.pcd_align = PCD_Align(nf=nf, groups=groups)
        if self.w_TSA:
            if self.align_target:
                self.tsa_fusion = TSA_Fusion(nf=nf, nframes=nframes, center=self.center)
            else:
                self.tsa_fusion = TSA_Fusion(nf=nf, nframes=nframes - 1, center=self.center)
        else:
            if self.align_target:
                self.tsa_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)
            else:
                self.tsa_fusion = nn.Conv2d((nframes - 1) * nf, nf, 1, 1, bias=True)
        #### reconstruction
        self.recon_trunk = arch_util.make_layer(ResidualBlock_noBN_f, back_RBs)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    
    def forward(self, x, scale):
        B, N, C, H, W = x.shape
        x = x.view(-1, C, H, W)
        x = F.interpolate(x, scale_factor=scale, mode='bilinear', 
                            align_corners=False)
        H, W = x.shape[-2:]
        x = x.view(B, N, C, H, W)
        x_center = x[:, self.center, :, :, :].contiguous()

        # L1
        if self.down_scale:
            L1_fea = self.lrelu(self.conv_first_1(x.view(-1, C, H, W)))
            L1_fea = self.lrelu(self.conv_first_2(L1_fea))
            L1_fea = self.lrelu(self.conv_first_3(L1_fea))
            H, W = H // 4, W // 4
        else:
            L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))  # BN,64,64,64
        L1_fea = self.feature_extraction(L1_fea)   # BN,64,64,64
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))  # BN,64,32,32
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))  # BN,64,32,32
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))  # BN,64,16,16
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))  # BN,64,16,16

        L1_fea = L1_fea.view(B, N, -1, H, W)            # B,N,64,64,64
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)  # B,N,64,32,32
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)  # B,N,64,16,16

        #### pcd align
        # ref feature list
        ref_fea_l = [L1_fea[:, self.center, :, :, :].clone(), 
                     L2_fea[:, self.center, :, :, :].clone(),
                     L3_fea[:, self.center, :, :, :].clone()]  
                   # [(B,64,64,64), (B,64,32,32), (B,64,16,16)]
        if self.ret_valid:
            aligned_fea, valids = [], []
            for i in range(N):
                if not self.align_target and (i == self.center): continue
                nbr_fea_l = [L1_fea[:, i, :, :, :].clone(), 
                             L2_fea[:, i, :, :, :].clone(),
                             L3_fea[:, i, :, :, :].clone()]
                feat, valid = self.pcd_align(nbr_fea_l, ref_fea_l)
                aligned_fea.append(feat)
                valids.append(valid)
        else:
            aligned_fea = []
            for i in range(N):
                if not self.align_target and (i == self.center): continue
                nbr_fea_l = [L1_fea[:, i, :, :, :].clone(), 
                             L2_fea[:, i, :, :, :].clone(),
                             L3_fea[:, i, :, :, :].clone()]
                feat = self.pcd_align(nbr_fea_l, ref_fea_l)
                aligned_fea.append(feat)
        aligned_fea = torch.stack(aligned_fea, dim=1)  # [B, N-1, C, H, W]

        if not self.w_TSA:
            aligned_fea = aligned_fea.view(B, -1, H, W)  # B,NC,64,64
        fea = self.tsa_fusion(aligned_fea)   # B,C,64,64

        out = self.recon_trunk(fea)   # B,C,64,64
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))  # B,C,128,128
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))  # B,C,256,256
        out = self.lrelu(self.HRconv(out))    # B,C,256,256
        out = self.conv_last(out)    # B,3,256,256
        out += x_center
        if not self.training: return out
        if self.ret_valid:
            return out, valid
        else:
            return out

class Scale2Weight(nn.Module):
    def __init__(self, kernel_size=3, inC=64, outC=64, dims=[8]):
        super(Scale2Weight,self).__init__()
        self.kernel_size=kernel_size
        self.inC = inC
        self.outC = outC
        self.dims = dims
        fcs = []
        in_dim = 1
        for dim in self.dims:
            fcs.append(nn.Linear(in_dim, dim))
            fcs.append(nn.ReLU(inplace=True))
            in_dim = dim
        fcs.append(nn.Linear(in_dim, self.kernel_size*self.kernel_size*self.inC*self.outC))
        self.scale_block=nn.Sequential(*fcs)

    def forward(self, input, scale):
        scale = input.new([scale])
        weight = self.scale_block(scale)
        weight = weight.view(self.outC, self.inC, self.kernel_size, self.kernel_size)
        output = F.conv2d(input, weight, padding=1)
        return output

class UPControlEDVR(nn.Module):
    def __init__(self, nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10, 
        center=None, w_TSA=True, down_scale=True, align_target=True, 
        ret_valid=False, multi_scale_cont=False):
        super(UPControlEDVR, self).__init__()
        self.nf = nf
        self.nframes = nframes
        self.center = nframes // 2 if center is None else center
        self.w_TSA = w_TSA
        self.down_scale = down_scale
        self.align_target = align_target
        self.ret_valid = ret_valid
        self.multi_scale_cont = multi_scale_cont
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)

        #### extract features (for each frame)
        if self.down_scale:
            self.conv_first_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
            self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
            self.conv_first_3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        else:
            self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.control_conv = Scale2Weight(kernel_size=3, inC=nf, outC=nf)
        if self.multi_scale_cont:
            self.control_conv2 = Scale2Weight(kernel_size=3, inC=nf, outC=nf)
            self.control_conv3 = Scale2Weight(kernel_size=3, inC=64, outC=64)

        if self.ret_valid:
            self.pcd_align = PCD_Align_Valid(nf=nf, groups=groups)
        else:
            self.pcd_align = PCD_Align(nf=nf, groups=groups)

        if self.w_TSA:
            self.tsa_fusion = TSA_Fusion(nf=nf, nframes=nframes, center=self.center)
        else:
            if self.align_target:
                self.tsa_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)
            else:
                self.tsa_fusion = nn.Conv2d((nframes - 1) * nf, nf, 1, 1, bias=True)

        #### reconstruction
        self.recon_trunk = arch_util.make_layer(ResidualBlock_noBN_f, back_RBs)
        #### upsampling
        if self.down_scale:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
            if self.multi_scale_cont:
                self.cont_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
                self.cont_conv3 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        else:
            self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nf, 64, 3, 1, 1, bias=True)
        # self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    
    def forward(self, x, scale):
        B, N, C, H, W = x.shape
        x = x.view(-1, C, H, W)
        x = F.interpolate(x, scale_factor=scale, mode='bilinear', 
                            align_corners=False)
        H, W = x.shape[-2:]
        x = x.view(B, N, C, H, W)
        x_center = x[:, self.center, :, :, :].contiguous()

        # L1
        if self.down_scale:
            L1_fea = self.lrelu(self.conv_first_1(x.view(-1, C, H, W)))
            L1_fea = self.lrelu(self.conv_first_2(L1_fea))
            L1_fea = self.lrelu(self.conv_first_3(L1_fea))
            H, W = H // 4, W // 4
        else:
            L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))
        L1_fea = self.feature_extraction(L1_fea)   # BN,64,64,64
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))  # BN,64,32,32
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))  # BN,64,32,32
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))  # BN,64,16,16
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))  # BN,64,16,16

        L1_fea = L1_fea.view(B, N, -1, H, W)            # B,N,64,64,64
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)  # B,N,64,32,32
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)  # B,N,64,16,16

        #### pcd align
        # ref feature list
        ref_fea_l = [L1_fea[:, self.center, :, :, :].clone(), 
                     L2_fea[:, self.center, :, :, :].clone(),
                     L3_fea[:, self.center, :, :, :].clone()]
                  # [(B,64,64,64), (B,64,32,32), (B,64,16,16)]
        if self.ret_valid:
            aligned_fea, valids = [], []
            for i in range(N):
                if not self.align_target and (i == self.center): continue
                nbr_fea_l = [L1_fea[:, i, :, :, :].clone(), 
                             L2_fea[:, i, :, :, :].clone(),
                             L3_fea[:, i, :, :, :].clone()]
                feat, valid = self.pcd_align(nbr_fea_l, ref_fea_l)
                aligned_fea.append(feat)
                valids.append(valid)
        else:
            aligned_fea = []
            for i in range(N):
                if not self.align_target and (i == self.center): continue
                nbr_fea_l = [L1_fea[:, i, :, :, :].clone(), 
                             L2_fea[:, i, :, :, :].clone(),
                             L3_fea[:, i, :, :, :].clone()]
                feat = self.pcd_align(nbr_fea_l, ref_fea_l)
                aligned_fea.append(feat)
        aligned_fea = torch.stack(aligned_fea, dim=1)

        if not self.w_TSA:
            aligned_fea = aligned_fea.view(B, -1, H, W)
        fea = self.tsa_fusion(aligned_fea)

        out = self.control_conv(fea, scale)
        out = self.recon_trunk(out)
        if self.down_scale:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            if self.multi_scale_cont:
                out = self.control_conv2(out, scale)
                out = self.lrelu(self.cont_conv2(out))
                out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
                out = self.control_conv3(out, scale)
                out = self.lrelu(self.cont_conv3(out))
            else:
                out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        else:
            out = self.lrelu(self.upconv1(out))
            out = self.lrelu(self.upconv2(out))
        out = self.lrelu(self.HRconv(out))
        out = self.conv_last(out)
        out += x_center
        if self.ret_valid:
            return out, valid
        else:
            return out

class FlowUPControlEDVR(nn.Module):
    def __init__(self, nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10, 
        center=None, w_TSA=True, down_scale=True, align_target=True, 
        ret_valid=False, multi_scale_cont=False):
        super(FlowUPControlEDVR, self).__init__()
        self.nf = nf
        self.nframes = nframes
        self.center = nframes // 2 if center is None else center
        self.w_TSA = w_TSA
        self.down_scale = down_scale
        self.align_target = align_target
        self.ret_valid = ret_valid
        self.multi_scale_cont = multi_scale_cont
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)

        #### extract features (for each frame)
        if self.down_scale:
            self.conv_first_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
            self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
            self.conv_first_3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        else:
            self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.control_conv = Scale2Weight(kernel_size=3, inC=nf, outC=nf)
        if self.multi_scale_cont:
            self.control_conv2 = Scale2Weight(kernel_size=3, inC=nf, outC=nf)
            self.control_conv3 = Scale2Weight(kernel_size=3, inC=64, outC=64)

        if self.ret_valid:
            self.pcd_align = PCD_Align_Valid(nf=nf, groups=groups)
        else:
            self.pcd_align = PCD_Align(nf=nf, groups=groups)

        if self.w_TSA:
            self.tsa_fusion = TSA_Fusion(nf=nf, nframes=nframes, center=self.center)
        else:
            if self.align_target:
                self.tsa_fusion = nn.Conv2d((nframes + 1) * nf, nf, 1, 1, bias=True)
            else:
                self.tsa_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)

        #### reconstruction
        self.recon_trunk = arch_util.make_layer(ResidualBlock_noBN_f, back_RBs)
        #### upsampling
        if self.down_scale:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
            if self.multi_scale_cont:
                self.cont_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
                self.cont_conv3 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        else:
            self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nf, 64, 3, 1, 1, bias=True)
        # self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    
    def forward(self, x, y, scale):
        B, N, C, H, W = x.shape
        x = x.view(-1, C, H, W)
        x = F.interpolate(x, scale_factor=scale, mode='bilinear', 
                            align_corners=False)
        H, W = x.shape[-2:]
        x = x.view(B, N, C, H, W)
        x_center = x[:, self.center, :, :, :].contiguous()

        # L1
        if self.down_scale:
            # LQs feature
            L1_fea = self.lrelu(self.conv_first_1(x.view(-1, C, H, W)))
            L1_fea = self.lrelu(self.conv_first_2(L1_fea))
            L1_fea = self.lrelu(self.conv_first_3(L1_fea))
            # frame(t-1) feature
            y1_fea = self.lrelu(self.conv_first_1(y))
            y1_fea = self.lrelu(self.conv_first_2(y1_fea))
            y1_fea = self.lrelu(self.conv_first_3(y1_fea))
            H, W = H // 4, W // 4
        else:
            L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))
            y1_fea = self.lrelu(self.conv_first(y))
        # L1
        L1_fea = self.feature_extraction(L1_fea)   # BN,64,64,64
        y_fea = self.feature_extraction(y1_fea)   # B,64,64,64
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))  # BN,64,32,32
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))  # BN,64,32,32
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))  # BN,64,16,16
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))  # BN,64,16,16

        L1_fea = L1_fea.view(B, N, -1, H, W)            # B,N,64,64,64
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)  # B,N,64,32,32
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)  # B,N,64,16,16

        #### pcd align
        # ref feature list
        ref_fea_l = [L1_fea[:, self.center, :, :, :].clone(), 
                     L2_fea[:, self.center, :, :, :].clone(),
                     L3_fea[:, self.center, :, :, :].clone()]
                  # [(B,64,64,64), (B,64,32,32), (B,64,16,16)]
        if self.ret_valid:
            aligned_fea, valids = [], []
            for i in range(N):
                if not self.align_target and (i == self.center): continue
                nbr_fea_l = [L1_fea[:, i, :, :, :].clone(), 
                             L2_fea[:, i, :, :, :].clone(),
                             L3_fea[:, i, :, :, :].clone()]
                feat, valid = self.pcd_align(nbr_fea_l, ref_fea_l)
                aligned_fea.append(feat)
                valids.append(valid)
        else:
            aligned_fea = []
            for i in range(N):
                if not self.align_target and (i == self.center): continue
                nbr_fea_l = [L1_fea[:, i, :, :, :].clone(), 
                             L2_fea[:, i, :, :, :].clone(),
                             L3_fea[:, i, :, :, :].clone()]
                feat = self.pcd_align(nbr_fea_l, ref_fea_l)
                aligned_fea.append(feat)
        aligned_fea.append(y_fea)
        aligned_fea = torch.stack(aligned_fea, dim=1)

        if not self.w_TSA:
            aligned_fea = aligned_fea.view(B, -1, H, W)
        fea = self.tsa_fusion(aligned_fea)

        out = self.control_conv(fea, scale)
        out = self.recon_trunk(out)
        if self.down_scale:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            if self.multi_scale_cont:
                out = self.control_conv2(out, scale)
                out = self.lrelu(self.cont_conv2(out))
                out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
                out = self.control_conv3(out, scale)
                out = self.lrelu(self.cont_conv3(out))
            else:
                out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        else:
            out = self.lrelu(self.upconv1(out))
            out = self.lrelu(self.upconv2(out))
        out = self.lrelu(self.HRconv(out))
        out = self.conv_last(out)
        out += x_center
        if self.ret_valid:
            return out, valid
        else:
            return out

class MultiEDVR(nn.Module):
    def __init__(self, nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10, center=None,
                 predeblur=False, HR_in=False, w_TSA=True):
        super(MultiEDVR, self).__init__()
        self.nf = nf
        self.nframes = nframes
        self.center = nframes // 2 if center is None else center
        self.is_predeblur = True if predeblur else False
        self.HR_in = True if HR_in else False
        self.w_TSA = w_TSA
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)

        #### extract features (for each frame)
        if self.is_predeblur:
            self.pre_deblur = Predeblur_ResNet_Pyramid(nf=nf, HR_in=self.HR_in)
            self.conv_1x1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        else:
            if self.HR_in:
                self.conv_first_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
                self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
                self.conv_first_3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
            else:
                self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.pcd_align = PCD_Align(nf=nf, groups=groups)
        if self.w_TSA:
            self.tsa_fusion = TSA_Fusion(nf=nf, nframes=nframes, center=self.center)
        else:
            self.tsa_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)

        #### reconstruction
        self.recon_trunk = arch_util.make_layer(ResidualBlock_noBN_f, back_RBs)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        B, N, C, H, W = x.size()  # N video frames
        x_center = x[:, self.center:N-self.center, :, :, :].contiguous() #B,K,C,H,W

        #### extract LR features
        # L1
        if self.is_predeblur:
            L1_fea = self.pre_deblur(x.view(-1, C, H, W))
            L1_fea = self.conv_1x1(L1_fea)
            if self.HR_in:
                H, W = H // 4, W // 4
        else:
            if self.HR_in:
                L1_fea = self.lrelu(self.conv_first_1(x.view(-1, C, H, W)))
                L1_fea = self.lrelu(self.conv_first_2(L1_fea))
                L1_fea = self.lrelu(self.conv_first_3(L1_fea))
                H, W = H // 4, W // 4
            else:
                L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))  # BN,64,64,64
        L1_fea = self.feature_extraction(L1_fea)   # BN,64,64,64
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))  # BN,64,32,32
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))  # BN,64,32,32
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))  # BN,64,16,16
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))  # BN,64,16,16

        L1_fea = L1_fea.view(B, N, -1, H, W)            # B,N,64,64,64
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)  # B,N,64,32,32
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)  # B,N,64,16,16

        #### pcd align
        # ref feature list
        aligned_feas = []
        for i in range(self.center, N - self.center):
            ref_fea_l = [
                L1_fea[:, i, :, :, :].clone(), 
                L2_fea[:, i, :, :, :].clone(),
                L3_fea[:, i, :, :, :].clone()
                ]   # [(B,64,64,64), (B,64,32,32), (B,64,16,16)] * (N-2*center)
            aligned_fea = []
            start = i - self.center
            for j in range(start, start + self.nframes):
                nbr_fea_l = [
                    L1_fea[:, j, :, :, :].clone(), 
                    L2_fea[:, j, :, :, :].clone(),
                    L3_fea[:, j, :, :, :].clone()
                ]
                aligned_fea.append(self.pcd_align(nbr_fea_l, ref_fea_l))  # N,(B,64,64,64)
            aligned_fea = torch.stack(aligned_fea, dim=1)  # [B, N, 64, 64, 64]
            aligned_feas.append(aligned_fea)
        aligned_feas = torch.stack(aligned_feas, dim=1)

        if not self.w_TSA:
            aligned_feas = aligned_feas.view(B*aligned_feas.shape[1], -1, H, W)  # BK,N64,64,64
        fea = self.tsa_fusion(aligned_feas)   # BK,64,64,64

        out = self.recon_trunk(fea)   # BK,64,64,64
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))  # BK,64,128,128
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))  # BK,64,256,256
        out = self.lrelu(self.HRconv(out))    # BK,64,256,256
        out = self.conv_last(out)    # BK,3,256,256
        out_c, out_h, out_w = out.shape[1:]
        out = out.view(B, -1, out_c, out_h, out_w)
        if self.HR_in:
            base = x_center
        else:
            x_center = x_center.view(-1, C, H, W)
            base = F.interpolate(x_center, scale_factor=4, mode='bilinear', 
                        align_corners=False)    # B,3,256,256
            base = base.view(B, -1, C, out_h, out_w)
        out += base
        return out

# Meta 
class Pos2Weight(nn.Module):
    def __init__(self,inC, kernel_size=3, outC=3):
        super(Pos2Weight,self).__init__()
        self.inC = inC
        self.kernel_size=kernel_size
        self.outC = outC
        self.meta_block=nn.Sequential(
            nn.Linear(3,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,self.kernel_size*self.kernel_size*self.inC*self.outC)
        )
    def forward(self,x):

        output = self.meta_block(x)
        return output

class MetaEDVR(nn.Module):
    def __init__(self, nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10, center=None,
                 predeblur=False, HR_in=False, w_TSA=True, has_final_conv=False, 
                 fix_edvr=False):
        """
        EDVR based arbitrary scale SR.
        """
        super(MetaEDVR, self).__init__()
        self.nf = nf
        self.center = nframes // 2 if center is None else center
        self.is_predeblur = True if predeblur else False
        self.HR_in = True if HR_in else False
        self.w_TSA = w_TSA
        self.has_final_conv = has_final_conv
        self.fix_edvr = fix_edvr
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)

        #### extract features (for each frame)
        if self.is_predeblur:
            self.pre_deblur = Predeblur_ResNet_Pyramid(nf=nf, HR_in=self.HR_in)
            self.conv_1x1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        else:
            if self.HR_in:
                self.conv_first_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
                self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
                self.conv_first_3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
            else:
                self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.pcd_align = PCD_Align(nf=nf, groups=groups)
        if self.w_TSA:
            self.tsa_fusion = TSA_Fusion(nf=nf, nframes=nframes, center=self.center)
        else:
            self.tsa_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)

        #### reconstruction
        self.recon_trunk = arch_util.make_layer(ResidualBlock_noBN_f, back_RBs)
        ## position to weight
        self.P2W = Pos2Weight(inC=nf)

        # #### upsampling
        # self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        # self.upconv2 = nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True)
        # self.pixel_shuffle = nn.PixelShuffle(2)
        # self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        if self.has_final_conv:
            self.conv_final = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def train(self, mode=True):
        nn.Module.train(self, mode)
        if mode:
            if self.fix_edvr:
                for k, v in self.named_parameters():
                    if not ('P2W' in k or 'conv_final' in k):
                        v.requires_grad = False

    def meta_upsample(self, x, pos_mat, scale):
        scale_int = math.ceil(scale)
        N,C,H,W = x.size()
        up_x = x.view(N,C,H,1,W,1)
        up_x = torch.cat([up_x]*scale_int,3)
        up_x = torch.cat([up_x]*scale_int,5).permute(0,3,5,1,2,4)
        up_x = up_x.contiguous().view(-1, C, H, W)

        local_weight = self.P2W(pos_mat.view(pos_mat.size(1),-1))

        cols = F.unfold(up_x, 3, padding=1)    # cols: N*r*r,64*9,H*W
        if not self.training: torch.cuda.empty_cache()
        
        cols = cols.view(cols.size(0)//(scale_int**2), scale_int**2, cols.size(1), 
            cols.size(2), 1).permute(0, 1, 3, 4, 2).contiguous()   # cols: N, r*r, H*W, 1, 64*9
        if not self.training: torch.cuda.empty_cache()

        local_weight = local_weight.view(x.size(2), scale_int, x.size(3), scale_int, 
            -1, 3).permute(1,3,0,2,4,5).contiguous()   # local_weight: r,r,H,W,64*9,3
        if not self.training: torch.cuda.empty_cache()
        
        local_weight = local_weight.view(scale_int**2, x.size(2) * x.size(3), -1, 3).contiguous() # r*r,H*W,64*9,3
        if not self.training: torch.cuda.empty_cache()
        
        out = torch.matmul(cols, local_weight).permute(0, 1, 4, 2, 3)  # N,r*r,H*W,1,3 => N,r*r,3,H*W,1
        out = out.contiguous().view(x.size(0),scale_int,scale_int, 3, x.size(2), 
                x.size(3)).permute(0, 3, 4, 1, 5, 2)   # N,r,r,3,50,50 => N,3,H,r,W,r
        out = out.contiguous().view(x.size(0), 3, scale_int*x.size(2), scale_int*x.size(3)) # N,3,H*r,W*r
        return out

    def forward(self, x, scale, pos_mat, mask):
        B, N, C, H, W = x.size()  # N video frames
        x_center = x[:, self.center, :, :, :].contiguous()

        #### extract LR features
        # L1
        if self.is_predeblur:
            L1_fea = self.pre_deblur(x.view(-1, C, H, W))
            L1_fea = self.conv_1x1(L1_fea)
            if self.HR_in:
                H, W = H // 4, W // 4
        else:
            if self.HR_in:
                L1_fea = self.lrelu(self.conv_first_1(x.view(-1, C, H, W)))
                L1_fea = self.lrelu(self.conv_first_2(L1_fea))
                L1_fea = self.lrelu(self.conv_first_3(L1_fea))
                H, W = H // 4, W // 4
            else:
                L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))  # BN,64,64,64
        L1_fea = self.feature_extraction(L1_fea)   # BN,64,64,64
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))  # BN,64,32,32
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))  # BN,64,32,32
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))  # BN,64,16,16
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))  # BN,64,16,16

        L1_fea = L1_fea.view(B, N, -1, H, W)            # B,N,64,64,64
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)  # B,N,64,32,32
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)  # B,N,64,16,16

        #### pcd align
        # ref feature list
        ref_fea_l = [
            L1_fea[:, self.center, :, :, :].clone(), L2_fea[:, self.center, :, :, :].clone(),
            L3_fea[:, self.center, :, :, :].clone()
        ]   # [(B,64,64,64), (B,64,32,32), (B,64,16,16)]
        aligned_fea = []
        for i in range(N):
            nbr_fea_l = [
                L1_fea[:, i, :, :, :].clone(), L2_fea[:, i, :, :, :].clone(),
                L3_fea[:, i, :, :, :].clone()
            ]
            aligned_fea.append(self.pcd_align(nbr_fea_l, ref_fea_l))  # N,(B,64,64,64)
        aligned_fea = torch.stack(aligned_fea, dim=1)  # [B, N, C, H, W]

        if not self.w_TSA:
            aligned_fea = aligned_fea.view(B, -1, H, W)  # B,NC,64,64
        fea = self.tsa_fusion(aligned_fea)   # B,C,64,64

        out = self.recon_trunk(fea)   # B,C,64,64
        out = self.meta_upsample(out, pos_mat, scale)
        out = torch.masked_select(out, mask)
        out = out.contiguous().view(B, C, int(H*scale), int(W*scale))
        # out = self.lrelu(self.HRconv(out))    # B,C,256,256
        if self.has_final_conv:
            out = self.conv_final(out)    # B,3,256,256
        
        if self.HR_in:
            base = x_center
        else:
            base = F.interpolate(x_center, scale_factor=scale, mode='bilinear', 
                        align_corners=False)    # B,3,256,256
        out += base
        return out

