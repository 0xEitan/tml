import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import consts
import defenses
import models
import utils

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

sns.set_theme()

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

torch.manual_seed(consts.SEED)
random.seed(consts.SEED)
np.random.seed(consts.SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_evaluation(sigma):
    # load model
    trained_models = {}
    mpath = f"trained-models/simple-cnn-sigma-{sigma:0.4f}"
    model = models.SimpleCNN()
    model.load_state_dict(torch.load(mpath))
    model.eval()
    model.to(device)

    # load test data
    data_test = utils.TMLDataset("test", transform=transforms.ToTensor())
    loader_test = DataLoader(data_test, batch_size=1, shuffle=True, num_workers=2)

    # init smoothed model
    smoothed_model = defenses.SmoothedModel(model, sigma)

    # find certified radius per sample
    cert_radii = []
    for x, y in tqdm(loader_test):
        x, y = x.to(device), y.to(device)
        pred, radius = smoothed_model.certify(
            x, consts.RS_N0, consts.RS_N, consts.RS_ALPHA, consts.BATCH_SIZE
        )
        cert_radii.append(radius if pred == y else 0.0)

    # done
    return cert_radii


def plot_radii(radii):
    x = []  # radius
    y = []  # accuracy

    # derive x and y from the certified radii
    for radius in radii:
        if radius == 0:
            continue

        x.append(radius)
        count = len(list(r for r in radii if r >= radius))
        y.append(count / len(radii))

    # plot
    plt.plot(x, y)


if __name__ == "__main__":
    sigmas = [0.05, 0.20]
    radii = {}
    for sigma in sigmas:
        print(f"Certifying L2 radii with sigma={sigma:0.4f}")
        radii[sigma] = run_evaluation(sigma)

    # plot
    plt.figure()
    for sigma in sigmas:
        plot_radii(radii[sigma])
    plt.xlabel("certified L2 radius")
    plt.ylabel("accuracy")
    plt.legend(sigmas, title="sigma")
    plt.savefig("randomized-smoothing-acc-vs-radius.pdf")
