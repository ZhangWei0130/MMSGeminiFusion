import torch
import torch.nn as nn
import math
from torch import Tensor
from torch.nn import functional as F
from torch.cuda.amp import autocast
from torch.utils.checkpoint import checkpoint
from semseg.models.backbones import *
from semseg.models.base import BaseModel
from semseg.models.heads import SegFormerHead
from semseg.models.layers import trunc_normal_

class HALFusionOptimized(nn.Module):
    def __init__(
        self,
        backbone_cfg: str = "HALF-B0",
        num_classes: int = 25,
        modals: list = ["img", "depth", "event", "lidar"],
        drop_path_rate: float = 0.0,
        use_checkpoint: bool = True,
        use_amp: bool = True
    ) -> None:
        super().__init__()
        self.modals = modals
        self.use_checkpoint = use_checkpoint
        self.use_amp = use_amp

        # 统一Backbone初始化
        self.backbones = nn.ModuleDict({
            'half': self._build_backbone(backbone_cfg, modals, drop_path_rate),
            'half_att': self._build_backbone("HALF_ATT-B0", modals, drop_path_rate)
        })

        # 动态解码头生成
        self.heads = nn.ModuleDict()
        for name, backbone in self.backbones.items():
            head_dim = 512 if 'B2' in backbone_cfg else 256
            self.heads[name] = SegFormerHead(
                backbone.embed_dims,
                head_dim,
                num_classes
            )

        # 可学习参数优化
        self.alpha = nn.Parameter(torch.ones(2))
        self._init_weights()

    def _build_backbone(self, cfg: str, modals: list, drop_path: float):
        """统一Backbone构造方法"""
        name, variant = cfg.split('-')
        return eval(name)(
            variant=variant,
            modals=modals,
            drop_path_rate=drop_path,
            num_modal=len(modals)
        )

    @autocast(enabled=True)
    def forward(self, x: list) -> list:
        # 并行特征提取
        features = {}
        for name in ['half_att', 'half']:
            backbone = self.backbones[name]
            if self.use_checkpoint and self.training:
                features[name] = checkpoint(backbone, x)
            else:
                features[name] = backbone(x)

        # 多分支处理
        outs = []
        # 处理half_att的两个分支
        for idx in range(2):
            out = self._process_branch(features['half_att'][idx], 'half_att')
            outs.append(out)
        
        # 处理half分支
        out_half = self._process_branch(features['half'], 'half')
        outs.append(out_half)

        # 动态融合（保留完整梯度流）
        alpha = F.softmax(self.alpha, dim=0)
        fused = sum(alpha[i] * outs[i] for i in range(2))  # 移除了detach()
        outs.append(fused)
        
        return [outs[-1]] if not self.training else outs

    def _process_branch(self, features, head_name):
        """统一分支处理逻辑"""
        out = self.heads[head_name](features)
        return F.interpolate(
            out, 
            size=features.shape[2:] if self.training else self.input_shape,
            mode='bilinear', 
            align_corners=False
        )

    def _init_weights(self):
        """优化的参数初始化"""
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
        nn.init.ones_(self.alpha)  # 初始化融合权重

    def init_pretrained(self, pretrained: str = None):
        """改进的预训练权重加载"""
        if pretrained:
            checkpoint = torch.load(pretrained, map_location="cpu")
            state_dict = checkpoint.get('model', checkpoint)
            
            # 自动适配不同权重格式
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('backbone.'):
                    new_state_dict[k[9:]] = v  # 去除backbone.前缀
                else:
                    new_state_dict[k] = v
            
            # 并行加载机制
            load_results = {}
            for name in ['half', 'half_att']:
                filtered = {k: v for k, v in new_state_dict.items() 
                          if k.startswith(self.backbones[name].prefix)}
                load_results[name] = self.backbones[name].load_state_dict(
                    self._adapt_weights(filtered, name), 
                    strict=False
                )
            print(f"Backbone half load: {load_results['half']}")
            print(f"Backbone half_att load: {load_results['half_att']}")

    def _adapt_weights(self, state_dict, branch_name):
        """权重适配转换"""
        converted = {}
        prefix_len = len(self.backbones[branch_name].prefix)
        for k, v in state_dict.items():
            new_key = k[prefix_len:]  # 去除分支特定前缀
            converted[new_key] = v
        return converted

if __name__ == "__main__":
    # 测试用例
    model = HALFusionOptimized(
        backbone_cfg="HALF-B2",
        num_classes=25,
        modals=["img"],
        use_checkpoint=True
    )
    model.init_pretrained("checkpoints/pretrained/segformer/mit_b2.pth")
    
    with torch.no_grad():
        x = [torch.randn(2, 3, 512, 512)]
        outputs = model(x)
        print(f"Output shapes: {[o.shape for o in outputs]}")