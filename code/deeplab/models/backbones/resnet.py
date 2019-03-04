import math
import torch.nn as nn

from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models.resnet import BasicBlock, Bottleneck


class ResNetBackbone(nn.Module):

    def __init__(self, resnet):
        super(ResNetBackbone, self).__init__()
        self.resnet = resnet

        self.out_planes = 512
        self.low_level_out_planes = 64        
        if isinstance(resnet.layer4[0], DilatedBottleneck):
            self.out_planes *= DilatedBottleneck.expansion
            self.low_level_out_planes *= DilatedBottleneck.expansion

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        z = x
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        return x, z        


class DilatedBasicBlock(BasicBlock):

    def __init__(self, inplanes, planes, stride=1, dilation=1, padding=1, downsample=None):
        super(DilatedBasicBlock, self).__init__(inplanes, planes, stride=stride, downsample=downsample)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, 
                               stride=stride, padding=padding, dilation=dilation,
                               bias=False)

class DilatedBottleneck(Bottleneck):

    def __init__(self, inplanes, planes, stride=1, dilation=1, padding=1, downsample=None):
        super(DilatedBottleneck, self).__init__(inplanes, planes, stride=stride, downsample=downsample)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, 
                               stride=stride, padding=padding, dilation=dilation,
                               bias=False)

def _build_layer3(block, in_planes, planes, n_blocks, stride, dilation):
    
    downsample = None
    if stride != 1 or in_planes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(in_planes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion)
        )

    layers = [block(in_planes, planes, stride=stride, 
                    dilation=dilation, padding=dilation,
                    downsample=downsample)]
    in_planes = planes * block.expansion
    for i in range(1, n_blocks):
        layers.append(block(in_planes, planes, stride=1, 
                            dilation=dilation, padding=dilation))

    return nn.Sequential(*layers)


def _build_layer4(block, in_planes, planes, stride, dilation, factors):
    
    downsample = None
    if stride != 1 or in_planes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(in_planes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion)
        )

    layers = [block(in_planes, planes, stride=stride, 
                    dilation=factors[0] * dilation, padding=factors[0] * dilation,
                    downsample=downsample)]
    in_planes = planes * block.expansion
    for i in range(1, len(factors)):
        layers.append(block(in_planes, planes, stride=1, 
                            dilation=factors[i] * dilation, padding=factors[i] * dilation))

    return nn.Sequential(*layers)


def build_backbone(resnet, output_stride):
    """
    Build backbone for DeepLabV3 model from a resnet model.    
    """
    assert isinstance(resnet, nn.Module)

    if isinstance(resnet.layer4[0], Bottleneck):
        block = DilatedBottleneck
        in_planes3 = 512
        in_planes4 = 1024
    else:
        block = DilatedBasicBlock
        in_planes3 = 128
        in_planes4 = 256

    # initialize weights of layer 3/4
    def _init_weight(modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    blocks = [1, 2, 4]
    if output_stride == 16:
        strides = [1, 2, 2, 1]
        dilations = [1, 1, 1, 2]

        resnet.layer4 = _build_layer4(block, in_planes=in_planes4, planes=512, 
                                      stride=strides[3], dilation=dilations[3], factors=blocks)

        _init_weight(resnet.layer4.modules())
    elif output_stride == 8:
        strides = [1, 2, 1, 1]
        dilations = [1, 1, 2, 4]

        resnet.layer3 = _build_layer3(block, in_planes=in_planes3, planes=256, n_blocks=len(resnet.layer3),
                                      stride=strides[2], dilation=dilations[2])

        resnet.layer4 = _build_layer4(block, in_planes=in_planes4, planes=512, 
                                      stride=strides[3], dilation=dilations[3], factors=blocks)
        
        _init_weight(resnet.layer3.modules())
        _init_weight(resnet.layer4.modules())
    else:
        raise NotImplementedError

    return ResNetBackbone(resnet)


def build_resnet18_backbone(output_stride, pretrained=True):
    model = resnet18(pretrained=pretrained)
    return build_backbone(model, output_stride=output_stride)


def build_resnet34_backbone(output_stride, pretrained=True):
    model = resnet34(pretrained=pretrained)
    return build_backbone(model, output_stride=output_stride)


def build_resnet50_backbone(output_stride, pretrained=True):
    model = resnet50(pretrained=pretrained)
    return build_backbone(model, output_stride=output_stride)


def build_resnet101_backbone(output_stride, pretrained=True):
    model = resnet101(pretrained=pretrained)
    return build_backbone(model, output_stride=output_stride)


if __name__ == "__main__":
    import torch
    
    x = torch.rand(2, 3, 360, 360)

    r18_backbone = build_resnet18_backbone(output_stride=16, pretrained=False)
    y, z = r18_backbone(x)
    print(y.shape, z.shape)

    r34_backbone = build_resnet34_backbone(output_stride=16, pretrained=False)
    y, z = r34_backbone(x)
    print(y.shape, z.shape)

    r50_backbone = build_resnet50_backbone(output_stride=16, pretrained=False)
    y, z = r50_backbone(x)
    print(y.shape, z.shape)

    r18_backbone = build_resnet18_backbone(output_stride=8, pretrained=False)
    y, z = r18_backbone(x)
    print(y.shape, z.shape)

    r34_backbone = build_resnet34_backbone(output_stride=8, pretrained=False)
    y, z = r34_backbone(x)
    print(y.shape, z.shape)

    r50_backbone = build_resnet50_backbone(output_stride=8, pretrained=False)
    y, z = r50_backbone(x)
    print(y.shape, z.shape)
