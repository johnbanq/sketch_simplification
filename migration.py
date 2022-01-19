import pickle

import torch


def migrate_to_all():
    model = torch.load("model_gan.pth")
    with open("model_gan.t7.extra.pickle", "rb") as f:
        cache = pickle.load(f)
    cache["model"] = model
    torch.save(cache, "model_gan.all.pth")

    model = torch.load("model_mse.pth")
    with open("model_mse.t7.extra.pickle", "rb") as f:
        cache = pickle.load(f)
    cache["model"] = model
    torch.save(cache, "model_mse.all.pth")


def add_type_to_pth():
    d = torch.load("model_gan.all.pth")
    d["type"] = "gan"
    torch.save(d, "model_gan.all.pth")
    d = torch.load("model_mse.all.pth")
    d["type"] = "mse"
    torch.save(d, "model_mse.all.pth")


if __name__ == '__main__':
    add_type_to_pth()
