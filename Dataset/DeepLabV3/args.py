from typing import Dict, List, Optional, Union
from .code import Bottleneck, FCNHead, ResNet, DeepLabHead

from torch import nn


backbones: Dict[str, List[nn.Module]] = {
    "ResNet50": [ResNet(Bottleneck, [3, 4, 6, 3],num_classes=100 , replace_stride_with_dilation=[False, True, True]),DeepLabHead(2048,100),None],
    "ResNet101": [ResNet(Bottleneck, [3, 4, 23, 3], replace_stride_with_dilation=[False, True, True]),DeepLabHead(2048,100),None],
}
args = [*backbones["ResNet50"]]