import argparse
import copy
from pathlib import Path
from typing import List

import PIL
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.transforms import Normalize, ToTensor
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


class ReplicationPadToMultipleOf8:

    def __init__(self, w: int, h: int):
        self.pw = 8 - (w % 8) if w % 8 != 0 else 0
        self.ph = 8 - (h % 8) if h % 8 != 0 else 0

    def __call__(self, data: Tensor) -> Tensor:
        if self.pw != 0 or self.ph != 0:
            return torch.nn.ReplicationPad2d((0, self.pw, 0, self.ph))(data)

    def __repr__(self):
        return self.__class__.__name__ + f"(pw={self.pw},ph={self.ph})"


class SimplificationPreprocessor:

    def __init__(self, w: int, h: int, image_mean: float, image_std: float):
        self.w, self.h = w, h
        self.to_tensor = ToTensor()
        self.normalize = Normalize(mean=image_mean, std=image_std)
        self.pad = ReplicationPadToMultipleOf8(w, h)

    def __call__(self, data: List[PIL.Image.Image]) -> Tensor:
        """
        preprocess a list of PIL images to format required by model
        """
        data = self._to_tensor(data)
        b, c, h, w = data.shape
        assert w == self.w and h == self.h
        data = self.normalize(data)
        data = self.pad(data)
        return data

    def _to_tensor(self, images: List[PIL.Image.Image]) -> Tensor:
        """
        mono-chrome-ify and stack the ToTensor result of PIL images
        """
        ret = []
        for img in images:
            tensor = self.to_tensor(img.convert('L'))
            ret.append(tensor)
        return torch.stack(ret)


use_cuda = torch.cuda.device_count() > 0


def main(opt):
    model = SimplificationModel.load_from_file(opt.model)
    if use_cuda:
        model = model.cuda()
    model.eval()

    data = PIL.Image.open(opt.img)
    width, height = data.size
    preprocess = SimplificationPreprocessor(
        w=width, h=height,
        image_mean=model.image_mean, image_std=model.image_std
    )
    data = preprocess([data])

    if use_cuda:
        data = data.cuda()
    pred = model.forward(data)
    if use_cuda:
        pred = pred.float()

    save_image(pred[0], opt.out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sketch simplification demo')
    parser.add_argument('--model', type=str, default='model_gan.all.pth', help='Model to use')
    parser.add_argument('--img', type=str, default='test.png', help='Input image file')
    parser.add_argument('--out', type=str, default='out.png', help='File to output')
    opt = parser.parse_args()

    main(opt)
