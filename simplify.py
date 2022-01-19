import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor
from torchvision import transforms
from torchvision.utils import save_image


class SimplificationModel(nn.Module):
    """
    represents the sketch simplification model
    """

    model: nn.Sequential
    image_mean: float
    image_std: float

    @staticmethod
    def load_from_file(file: Path):
        d = torch.load(file)
        if d["type"] == "gan":
            import model_gan
            model = copy.deepcopy(model_gan.model_gan)
        elif d["type"] == "mse":
            import model_mse
            model = copy.deepcopy(model_mse.model_mse)
        else:
            raise NotImplementedError()

        model.load_state_dict(d["model"])
        immean = d['mean']
        imstd = d['std']

        return SimplificationModel(model, mean=immean, std=imstd)

    def __init__(self, model: nn.Sequential, mean: float, std: float):
        super().__init__()
        self.model = model
        self.image_mean = mean
        self.image_std = std

    def forward(self, data: Tensor) -> Tensor:
        return self.model.forward(data)


use_cuda = torch.cuda.device_count() > 0


def main(opt):
    model = SimplificationModel.load_from_file(opt.model)
    model.eval()

    data = Image.open(opt.img).convert('L')
    w, h = data.size[0], data.size[1]
    pw = 8 - (w % 8) if w % 8 != 0 else 0
    ph = 8 - (h % 8) if h % 8 != 0 else 0
    data = ((transforms.ToTensor()(data) - model.image_mean) / model.image_std).unsqueeze(0)
    if pw != 0 or ph != 0:
        data = torch.nn.ReplicationPad2d((0, pw, 0, ph))(data).data

    if use_cuda:
        pred = model.cuda().forward(data.cuda()).float()
    else:
        pred = model.forward(data)
    save_image(pred[0], opt.out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sketch simplification demo')
    parser.add_argument('--model', type=str, default='model_gan.all.pth', help='Model to use')
    parser.add_argument('--img', type=str, default='test.png', help='Input image file')
    parser.add_argument('--out', type=str, default='out.png', help='File to output')
    opt = parser.parse_args()

    main(opt)
