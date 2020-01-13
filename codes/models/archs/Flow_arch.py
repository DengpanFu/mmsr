#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-01-13 15:04:22
# @Author  : Dengpan Fu (v-defu@microsoft.com)

"""PWC Net for extract optical flow """

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
try:
    from correlation import correlation
except Exception as exc:
    module_path = os.path.join(os.path.abspath(
        os.path.dirname(__file__)), 'correlation')
    sys.path.insert(0, module_path)
    import correlation

Backward_tensorGrid, Backward_tensorPartial = {}, {}

def Backward(tensorInput, tensorFlow):
    if str(tensorFlow.size()) not in Backward_tensorGrid:
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, 
            tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, 
            tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

        Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([tensorHorizontal, \
            tensorVertical], 1).to(tensorFlow.device)

    if str(tensorFlow.size()) not in Backward_tensorPartial:
        Backward_tensorPartial[str(tensorFlow.size())] = tensorFlow.new_ones(
            [tensorFlow.size(0), 1, tensorFlow.size(2), tensorFlow.size(3) ])

    tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), 
        tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)
    tensorInput = torch.cat([tensorInput, Backward_tensorPartial[str(tensorFlow.size())]], 1)

    tensorOutput = torch.nn.functional.grid_sample(input=tensorInput, 
                        grid=(Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow
                        ).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')

    tensorMask = tensorOutput[:, -1:, :, :]
    tensorMask[tensorMask > 0.999] = 1.0
    tensorMask[tensorMask < 1.0] = 0.0

    return tensorOutput[:, :-1, :, :] * tensorMask

class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()

        self.moduleOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=16, 
                            kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=16, out_channels=16, 
                            kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=16, out_channels=16, 
                            kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleTwo = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16, out_channels=32, 
                            kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=32, out_channels=32, 
                            kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=32, out_channels=32, 
                            kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleThr = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=64, 
                            kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=64, out_channels=64, 
                            kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=64, out_channels=64, 
                            kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleFou = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=96, 
                            kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=96, out_channels=96, 
                            kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=96, out_channels=96, 
                            kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleFiv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=96, out_channels=128, 
                            kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=128, 
                            kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=128, 
                            kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleSix = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=196, 
                            kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=196, out_channels=196, 
                            kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=196, out_channels=196, 
                            kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

    def forward(self, tensorInput):
        tensorOne = self.moduleOne(tensorInput)
        tensorTwo = self.moduleTwo(tensorOne)
        tensorThr = self.moduleThr(tensorTwo)
        tensorFou = self.moduleFou(tensorThr)
        tensorFiv = self.moduleFiv(tensorFou)
        tensorSix = self.moduleSix(tensorFiv)
        output = [tensorOne, tensorTwo, tensorThr, tensorFou, tensorFiv, tensorSix]
        return output

class Decoder(nn.Module):
    def __init__(self, intLevel):
        super(Decoder, self).__init__()

        intPrevious = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, \
                        81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 1]
        intCurrent = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, \
                        81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 0]

        if intLevel < 6: 
            self.moduleUpflow = torch.nn.ConvTranspose2d(in_channels=2, 
                out_channels=2, kernel_size=4, stride=2, padding=1)
        if intLevel < 6: 
            self.moduleUpfeat = torch.nn.ConvTranspose2d(
                in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, 
                out_channels=2, kernel_size=4, stride=2, padding=1)
        if intLevel < 6: 
            self.dblBackward = [ None, None, None, 5.0, 2.5, \
                                    1.25, 0.625, None ][intLevel + 1]

        self.moduleOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, 
                            kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleTwo = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, 
                            kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleThr = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, 
                            kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleFou = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, 
                            out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleFiv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, 
                            out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleSix = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, 
                            out_channels=2, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, tensorFirst, tensorSecond, objectPrevious):
        tensorFlow = None
        tensorFeat = None

        if objectPrevious is None:
            tensorFlow = None
            tensorFeat = None

            tensorVolume = torch.nn.functional.leaky_relu(
                    input=correlation.FunctionCorrelation(tensorFirst=tensorFirst, 
                    tensorSecond=tensorSecond), negative_slope=0.1, inplace=False)

            tensorFeat = torch.cat([ tensorVolume ], 1)

        elif objectPrevious is not None:
            tensorFlow = self.moduleUpflow(objectPrevious['tensorFlow'])
            tensorFeat = self.moduleUpfeat(objectPrevious['tensorFeat'])

            tensorVolume = torch.nn.functional.leaky_relu(
                    input=correlation.FunctionCorrelation(tensorFirst=tensorFirst, 
                    tensorSecond=Backward(tensorInput=tensorSecond, 
                        tensorFlow=tensorFlow * self.dblBackward)), 
                        negative_slope=0.1, inplace=False)

            tensorFeat = torch.cat([tensorVolume, tensorFirst, \
                            tensorFlow, tensorFeat], 1)

        tensorFeat = torch.cat([self.moduleOne(tensorFeat), tensorFeat], 1)
        tensorFeat = torch.cat([self.moduleTwo(tensorFeat), tensorFeat], 1)
        tensorFeat = torch.cat([self.moduleThr(tensorFeat), tensorFeat], 1)
        tensorFeat = torch.cat([self.moduleFou(tensorFeat), tensorFeat], 1)
        tensorFeat = torch.cat([self.moduleFiv(tensorFeat), tensorFeat], 1)
        tensorFlow = self.moduleSix(tensorFeat)

        return {
            'tensorFlow': tensorFlow,
            'tensorFeat': tensorFeat
        }

class Refiner(nn.Module):
    def __init__(self):
        super(Refiner, self).__init__()

        self.moduleMain = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, 
                    out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, 
                    stride=1, padding=2, dilation=2),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, 
                    stride=1, padding=4, dilation=4),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, 
                    stride=1, padding=8, dilation=8),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, 
                    stride=1, padding=16, dilation=16),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, 
                    stride=1, padding=1, dilation=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, 
                    stride=1, padding=1, dilation=1)
        )

    def forward(self, tensorInput):
        return self.moduleMain(tensorInput)

class PWCNet(nn.Module):
    def __init__(self):
        super(PWCNet, self).__init__()
        self.moduleExtractor = Extractor()

        self.moduleTwo = Decoder(2)
        self.moduleThr = Decoder(3)
        self.moduleFou = Decoder(4)
        self.moduleFiv = Decoder(5)
        self.moduleSix = Decoder(6)

        self.moduleRefiner = Refiner()

        # self.load_state_dict(torch.load(opt.model))

    def forward(self, tensorFirst, tensorSecond):
        tensorFirst = self.moduleExtractor(tensorFirst)
        tensorSecond = self.moduleExtractor(tensorSecond)

        objectEstimate = self.moduleSix(tensorFirst[-1], tensorSecond[-1], None)
        objectEstimate = self.moduleFiv(tensorFirst[-2], tensorSecond[-2], objectEstimate)
        objectEstimate = self.moduleFou(tensorFirst[-3], tensorSecond[-3], objectEstimate)
        objectEstimate = self.moduleThr(tensorFirst[-4], tensorSecond[-4], objectEstimate)
        objectEstimate = self.moduleTwo(tensorFirst[-5], tensorSecond[-5], objectEstimate)

        return objectEstimate['tensorFlow'] + self.moduleRefiner(objectEstimate['tensorFeat'])

def estimate_flow(net, first, second):
    assert(first.shape[-2:] == second.shape[-2:])

    H, W = first.shape[-2:]

    procH, procW = math.ceil(H / 64) * 64, math.ceil(W / 64) * 64

    procFirst = F.interpolate(input=first, size=(procH, procW), 
                              mode='bilinear', align_corners=False)
    procSecond = F.interpolate(input=second, size=(procH, procW), 
                              mode='bilinear', align_corners=False)
    with torch.no_grad():
        flow = net(procFirst, procSecond)
    flow = 20.0 * F.interpolate(input=flow, size=(H, W), 
                                mode='bilinear', align_corners=False)

    flow[:, 0, :, :] *= float(H) / float(procH)
    flow[:, 1, :, :] *= float(W) / float(procW)

    return flow

Mesh_Grid = {}
def wraping(x, flow):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    shape_str = str(x.shape)
    # mesh grid
    if shape_str not in Mesh_Grid:
        B, C, H, W = x.shape
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        grid = grid.to(x.device)
        Mesh_Grid[shape_str] = grid
    else:
        grid = Mesh_Grid[shape_str]
        H, W = x.shape[-2:]

    vgrid = grid + flow
    # scale grid to [-1, 1] 
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid)
    mask = x.new_ones(x.size())
    mask = F.grid_sample(mask, vgrid)

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output*mask, mask

def magnitude(x):
    return (x ** 2).sum(dim=1)

def compute_flow_gradients(x):
    du = x.new_zeros(x.shape)
    dv = x.new_zeros(x.shape)
    du[:, :, :, :-1] = x[:, :, :, :-1] - x[:, :, :, 1:]
    dv[:, :, :-1, :] = x[:, :, :-1, :] - x[:, :, 1:, :]
    return du, dv


def detect_occlusion(flow1, flow2):
    warped_flow1, _ = wraping(flow1, flow2)

    f1_f2_sum = warped_flow1 + flow2
    # occlusion
    mag_f12 = magnitude(f1_f2_sum)
    mag_wf1 = magnitude(warped_flow1)
    mag_f2 = magnitude(flow2)
    mask1 =  mag_f12 > 0.01 * (mag_wf1 + mag_f2) + 0.5
    # motion boundary
    du, dv = compute_flow_gradients(flow2)
    mask2 = (magnitude(du) + magnitude(dv)) > 0.01 * mag_f2 + 0.002
    mask = ~(mask1 | mask2)
    return mask.unsqueeze(1).float()



if __name__ == "__main__":
    x = torch.rand(1,2)
