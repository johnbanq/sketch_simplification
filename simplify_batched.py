import argparse
import copy
import math
from itertools import count
from pathlib import Path
from typing import List, Iterable

import PIL
import torch
import torch.nn as nn
from more_itertools import ichunked, chunked
from torch import Tensor
from torchvision.transforms import Normalize, ToTensor
from torchvision.utils import save_image
from tqdm import tqdm


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
        else:
            return data

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


def find_pictures(folder: Path) -> Iterable[Path]:
    """
    find pictures in folder
    """
    return folder.glob("*.jpg")


def count_pictures(folder: Path) -> int:
    """
    count pictures in folder
    """
    return sum(1 for _ in find_pictures(folder))


use_cuda = False


def main(args):
    model = SimplificationModel.load_from_file(args.model)
    if use_cuda:
        model = model.cuda()
    model = model.eval()

    # probe image size
    sample_pic = next(iter(find_pictures(args.folder)))
    sample_pic = PIL.Image.open(sample_pic)
    width, height = sample_pic.size
    sample_pic = None
    preprocess = SimplificationPreprocessor(
        w=width, h=height,
        image_mean=model.image_mean, image_std=model.image_std
    )

    # chunked processing
    pics = find_pictures(args.folder)
    pics_count = count_pictures(args.folder)
    batch_size = args.batch_size
    for path_chunk in tqdm(chunked(pics, batch_size), total=math.ceil(pics_count/batch_size), desc="processing..."):
        with torch.no_grad():
            data = preprocess([PIL.Image.open(i) for i in path_chunk])
            if use_cuda:
                data = data.cuda()
            pred = model.forward(data)
            if use_cuda:
                pred = pred.float()

        for p, i in zip(pred, path_chunk):
            i_out = i.parent / (i.name[:-len(".jpg")]+".processed.jpeg")
            save_image(p, i_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batched Sketch Simplification Processor')
    parser.add_argument('--folder', type=Path, help="folder to fetch jpg pictures from and to place into")
    parser.add_argument('--model', type=str, default='model_gan.all.pth', help='Model to use')
    parser.add_argument('--batch-size', type=int, default=1, help="amount of pics per batch")
    opt = parser.parse_args()

    main(opt)
