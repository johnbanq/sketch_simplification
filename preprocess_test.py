import unittest
from pathlib import Path

import PIL
import torch
from PIL.Image import Image
from torchvision.transforms import transforms

from simplify import SimplificationPreprocessor


def old_preprocessor(path, image_mean: float, image_std: float):
    data = PIL.Image.open(path).convert('L')
    w, h = data.size[0], data.size[1]
    pw = 8 - (w % 8) if w % 8 != 0 else 0
    ph = 8 - (h % 8) if h % 8 != 0 else 0
    data = ((transforms.ToTensor()(data) - image_mean) / image_std).unsqueeze(0)
    if pw != 0 or ph != 0:
        data = torch.nn.ReplicationPad2d((0, pw, 0, ph))(data).data
    return data


class PreprocessTest(unittest.TestCase):

    file: Path = Path("figs/fig01_eisaku.png")

    image_mean: float = 0.9664114577640158

    image_std: float = 0.0858381272736797

    def test_preprocessor_is_consistent_with_old(self):
        data = PIL.Image.open(self.file)
        width, height = data.size
        preprocess = SimplificationPreprocessor(
            w=width, h=height,
            image_mean=self.image_mean, image_std=self.image_std
        )
        result = preprocess([data])

        old_result = old_preprocessor(self.file, self.image_mean, self.image_std)

        torch.testing.assert_close(result, old_result)


if __name__ == '__main__':
    unittest.main()