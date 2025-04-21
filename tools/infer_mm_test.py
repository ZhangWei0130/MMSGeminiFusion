import torch
import argparse
import yaml
import math
from torch import Tensor
from torch.nn import functional as F
from pathlib import Path
from torchvision import io
from torchvision import transforms as T
import torchvision.transforms.functional as TF 
from semseg.models import *
from semseg.datasets import *
from semseg.utils.utils import timer
from semseg.utils.visualize import draw_text
import glob
import os
from PIL import Image, ImageDraw, ImageFont


class SemSeg:
    def __init__(self, cfg) -> None:
        self.device = torch.device(cfg['DEVICE'])

        self.palette = eval(cfg['DATASET']['NAME']).PALETTE
        self.labels = eval(cfg['DATASET']['NAME']).CLASSES

        self.model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], len(self.palette), cfg['DATASET']['MODALS'])
        msg = self.model.load_state_dict(torch.load(cfg['EVAL']['MODEL_PATH'], map_location='cpu'))
        print(msg)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.size = cfg['TEST']['IMAGE_SIZE']
        self.tf_pipeline_img = T.Compose([
            T.Resize(self.size),
            T.Lambda(lambda x: x / 255),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            T.Lambda(lambda x: x.unsqueeze(0))
        ])
        self.tf_pipeline_modal = T.Compose([
            T.Resize(self.size),
            T.Lambda(lambda x: x / 255),
            T.Lambda(lambda x: x.unsqueeze(0))
        ])

    def postprocess(self, orig_img: Tensor, seg_map: Tensor, overlay: bool) -> Tensor:
        seg_map = seg_map.softmax(dim=1).argmax(dim=1).cpu().to(int)
        seg_image = self.palette[seg_map].squeeze()
        if overlay: 
            seg_image = (orig_img.permute(1, 2, 0) * 0.4) + (seg_image * 0.6)
        image = seg_image.to(torch.uint8)
        return Image.fromarray(image.numpy())

    def postprocess_label(self, orig_img: Tensor, label: Tensor, overlay: bool) -> Tensor:
        label = label.squeeze().cpu().to(int)
        label_image = self.palette[label].squeeze()
        if overlay:
            label_image = (orig_img.permute(1, 2, 0) * 0.4) + (label_image * 0.6)
        image = label_image.to(torch.uint8)
        return Image.fromarray(image.numpy())

    @torch.inference_mode()
    @timer
    def model_forward(self, img: Tensor) -> Tensor:
        return self.model(img)
    
    def _open_img(self, file):
        img = io.read_image(file)
        C, H, W = img.shape
        if C == 4:
            img = img[:3, ...]
        if C == 1:
            img = img.repeat(3, 1, 1)
        return img

    def predict(self, img_fname: str, overlay: bool) -> tuple:
        if cfg['DATASET']['NAME'] == 'DELIVER':
            x1 = img_fname.replace('/img', '/hha').replace('_rgb', '_depth')
            x2 = img_fname.replace('/img', '/lidar').replace('_rgb', '_lidar')
            x3 = img_fname.replace('/img', '/event').replace('_rgb', '_event')
            lbl_path = img_fname.replace('/img', '/semantic').replace('_rgb', '_semantic')
        elif cfg['DATASET']['NAME'] == 'KITTI360':
            x1 = os.path.join(img_fname.replace('data_2d_raw', 'data_2d_hha'))
            x2 = os.path.join(img_fname.replace('data_2d_raw', 'data_2d_lidar'))
            x2 = x2.replace('.png', '_color.png')
            x3 = os.path.join(img_fname.replace('data_2d_raw', 'data_2d_event'))
            x3 = x3.replace('/image_00/data_rect/', '/').replace('.png', '_event_image.png')
            lbl_path = os.path.join(*[img_fname.replace('data_2d_raw', 'data_2d_semantics/train').replace('data_rect', 'semantic')])

        image = io.read_image(img_fname)[:3, ...]
        img = self.tf_pipeline_img(image).to(self.device)

        x1 = self._open_img(x1)
        x1 = self.tf_pipeline_modal(x1).to(self.device)
        x2 = self._open_img(x2)
        x2 = self.tf_pipeline_modal(x2).to(self.device)
        x3 = self._open_img(x3)
        x3 = self.tf_pipeline_modal(x3).to(self.device)

        label = io.read_image(lbl_path)
        label = label[0, ...].unsqueeze(0)
        label = T.Resize(self.size, interpolation=T.InterpolationMode.NEAREST)(label)
        label[label == 255] = 0
        label -= 1

        sample = [img, x1, x2, x3][:len(modals)]
        if 'real' in cfg["MODEL"]["ROOT"]:
            seg_map = self.model_forward(sample)[-2]
        else:
            seg_map = self.model_forward(sample)[-1]
        
        return (
            self.postprocess(image, seg_map, overlay),
            self.postprocess_label(image, label, overlay)
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/DELIVER.yaml')
    args = parser.parse_args()
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    cases = ['lidarjitter']
    modals = cfg['DATASET']['MODALS']

    test_file = Path(cfg['TEST']['FILE'])
    if not test_file.exists():
        raise FileNotFoundError(test_file)

    modals_name = ''.join([m[0] for m in cfg['DATASET']['MODALS']])
    save_dir = Path(cfg['SAVE_DIR']) / 'test_results' / (cfg['DATASET']['NAME']+'_'+cfg['MODEL']['BACKBONE']+'_'+modals_name)
    
    semseg = SemSeg(cfg)

    if test_file.is_file():
        segmap, labelmap = semseg.predict(str(test_file), cfg['TEST']['OVERLAY'])
        (save_dir/'predictions').mkdir(parents=True, exist_ok=True)
        (save_dir/'labels').mkdir(parents=True, exist_ok=True)
        segmap.save(save_dir/'predictions'/f"{test_file.stem}.png")
        labelmap.save(save_dir/'labels'/f"{test_file.stem}.png")
    else:
        if cfg['DATASET']['NAME'] == 'DELIVER':
            files = sorted(glob.glob(os.path.join(*[str(test_file), 'img', '*', 'val', '*', '*.png'])))
        elif cfg['DATASET']['NAME'] == 'KITTI360':
            source = os.path.join(test_file, 'val.txt')
            files = []
            with open(source) as f:
                files_ = f.readlines()
            for item in files_:
                file_name = item.strip()
                if ' ' in file_name:
                    file_name = os.path.join(*[str(test_file), file_name.split(' ')[0]])
                files.append(file_name)
        else:
            raise NotImplementedError()

        for file in files:
            segmap, labelmap = semseg.predict(file, cfg['TEST']['OVERLAY'])
            relative_path = Path(os.path.relpath(file, test_file))
            pred_save_path = save_dir/'predictions'/relative_path
            pred_save_path.parent.mkdir(parents=True, exist_ok=True)
            segmap.save(pred_save_path)
            
            label_save_path = save_dir/'labels'/relative_path
            label_save_path.parent.mkdir(parents=True, exist_ok=True)
            labelmap.save(label_save_path)
            
            print(f"Prediction saved: {pred_save_path}")
            print(f"Label saved: {label_save_path}")