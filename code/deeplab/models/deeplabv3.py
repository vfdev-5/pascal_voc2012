import torch
import torch.nn as nn
import torch.nn.functional as F

from models.aspp import build_aspp
from models.decoders import build_decoder


class DeepLabV3(nn.Module):

    def __init__(self, build_backbone_fn, output_stride=16, num_classes=21):
        super(DeepLabV3, self).__init__()

        self.backbone = build_backbone_fn(output_stride)
        assert hasattr(self.backbone, "out_planes")
        assert hasattr(self.backbone, "low_level_out_planes")
        self.aspp = build_aspp(self.backbone.out_planes, output_stride)
        self.decoder = build_decoder(num_classes, self.backbone.low_level_out_planes)
    
    def forward(self, x):
        input_size = x.shape[2:]
        x, z = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, z)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        return x    
    

if __name__ == "__main__":
    from functools import partial
    import torch

    from models.backbones import build_resnet34_backbone
    
    x = torch.rand(2, 3, 360, 360)

    r34_backbone_fn = partial(build_resnet34_backbone, pretrained=False)
    model = DeepLabV3(r34_backbone_fn, output_stride=16)
    y = model(x)
    print(y.shape)
