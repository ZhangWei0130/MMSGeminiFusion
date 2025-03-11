import torch
import torch.nn as nn
import math
from torch import Tensor
from torch.nn import functional as F
from semseg.models.backbones import *
from semseg.models.base import BaseModel
from semseg.models.heads import SegFormerHead
from semseg.models.layers import trunc_normal_

class MMSGeminiFusion(nn.Module):
    def __init__(
        self,
        backbone: str = "MMSGemini-B0",
        num_classes: int = 25,
        modals: list = ["img", "depth", "event", "lidar"],
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()

        backbone, variant = backbone.split("-")
        self.backbone_mms = eval(backbone)(
            variant,
            modals
        )
        self.modals = modals

        backbone = "GeminiFusionBackbone"
        self.backbone_gemini = eval(backbone)(
            variant,
            modals,
            drop_path_rate=drop_path_rate,
            num_modal=len(modals),
        )

        self.decode_head_mms = SegFormerHead(
            self.backbone_mms.channels, 
            256 if 'B0' in backbone or 'B1' in backbone else 512, 
            num_classes
        )

        self.decode_head_gemini = SegFormerHead(
            self.backbone_gemini.embed_dims,
            256 if "B0" in backbone or "B1" in backbone else 512,
            num_classes,
        )
        self.apply(self._init_weights)

        self.num_Gemini = 2
        self.num_parallel = 3
        self.alpha = torch.nn.Parameter(
            torch.ones(self.num_parallel, requires_grad=True)
        )
        self.register_parameter("alpha", self.alpha)

    def forward(self, x: list) -> list:
        x_mms = self.backbone_mms(x)
        x_gemini = self.backbone_gemini(x)
        outs = []
        out = self.decode_head_mms(x_mms)
        out = F.interpolate(out, size=x[0].shape[2:], mode='bilinear', align_corners=False)
        outs.append(out)

        for idx in range(self.num_Gemini):
            out = self.decode_head_gemini(x_gemini[idx])
            out = F.interpolate(
                out, size=x[0].shape[2:], mode='bilinear', align_corners=False
            )
            outs.append(out)
        ens = 0
        alpha_soft = F.softmax(self.alpha, dim=0)
        for idx in range(self.num_parallel):
            ens += alpha_soft[idx] * outs[idx].detach()
        outs.append(ens)
        return outs

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            checkpoint = torch.load(pretrained, map_location="cpu")
            if "state_dict" in checkpoint.keys():
                checkpoint = checkpoint["state_dict"]
            if "model" in checkpoint.keys():
                checkpoint = checkpoint["model"]
            checkpoint.pop("head.weight")
            checkpoint.pop("head.bias")
            checkpoint_mms = self._expand_state_dict(
                self.backbone_mms.state_dict(), checkpoint, self.num_parallel
            )
            checkpoint_gemini = self._expand_state_dict(
                self.backbone_gemini.state_dict(), checkpoint, self.num_parallel
            )
            msg = self.backbone_mms.load_state_dict(checkpoint_mms, strict=True)
            print(msg)
            msg = self.backbone_gemini.load_state_dict(checkpoint_gemini, strict=True)
            print(msg)


    def _expand_state_dict(self, model_dict, state_dict, num_parallel):
        model_dict_keys = model_dict.keys()
        state_dict_keys = state_dict.keys()
        for model_dict_key in model_dict_keys:
            model_dict_key_re = model_dict_key.replace("module.", "")
            if model_dict_key_re in state_dict_keys:
                model_dict[model_dict_key] = state_dict[model_dict_key_re]
            for i in range(num_parallel):
                ln = ".ln_%d" % i
                replace = True if ln in model_dict_key_re else False
                model_dict_key_re = model_dict_key_re.replace(ln, "")
                if replace and model_dict_key_re in state_dict_keys:
                    model_dict[model_dict_key] = state_dict[model_dict_key_re]
        return model_dict


if __name__ == "__main__":
    modals = ["img"]
    # modals = ['img', 'depth', 'event', 'lidar']
    model = MMSGemini("MMSGemini-B2", 25, modals)
    model.init_pretrained("checkpoints/pretrained/segformer/mit_b2.pth")
    x = [torch.zeros(1, 3, 512, 512)]
    y = model(x)
    print(y.shape)
