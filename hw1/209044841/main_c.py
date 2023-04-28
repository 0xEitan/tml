import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchvision.transforms as transforms

import consts
import models
import utils

sns.set_theme()

torch.manual_seed(consts.SEED)
random.seed(consts.SEED)
np.random.seed(consts.SEED)

# GPU available?
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load model and dataset
model = utils.load_pretrained_cnn(1).to(device)
model.eval()
dataset = utils.TMLDataset(transform=transforms.ToTensor())
data_loader = torch.utils.data.DataLoader(dataset, batch_size=consts.BATCH_SIZE)

# model accuracy
acc_orig = utils.compute_accuracy(model, data_loader, device)
print(f"Model accuracy before flipping: {acc_orig:0.4f}")  # 0.8250

# layers whose weights will be flipped
layers = {
    "conv1": model.conv1,
    "conv2": model.conv2,
    "fc1": model.fc1,
    "fc2": model.fc2,
    "fc3": model.fc3,
}

# flip bits at random and measure impact on accuracy (via RAD)
# will contain a list of RADs for each index of bit flipped
RADs_bf_idx = dict([(bf_idx, []) for bf_idx in range(32)])
RADs_all = []  # will eventually contain all consts.BF_PER_LAYER*len(layers) RADs
for layer_name in layers:
    layer = layers[layer_name]
    with torch.no_grad():
        W = layer.weight
        W.requires_grad = False
        for _ in range(consts.BF_PER_LAYER):
            w = W.clone().flatten()

            # pick a random float from W
            idx = np.random.randint(0, len(w))
            w_idx = w[idx]

            # flip a random bit from that float
            w_idx, bf_idx = utils.random_bit_flip(w_idx)
            w[idx] = w_idx

            # apply change
            w = w.reshape(layer.weight.shape)
            layer.weight = torch.nn.Parameter(w)

            # measure RAD
            acc_bf = utils.compute_accuracy(model, data_loader, device)
            rad = (acc_orig - acc_bf) / acc_orig

            # restore original weight
            layer.weight = W

            # store results
            RADs_bf_idx[bf_idx].append(rad)
            RADs_all.append(rad)

# Max and % RAD>10%
RADs_all = np.array(RADs_all)
print(f"Total # weights flipped: {len(RADs_all)}")
print(f"Max RAD: {np.max(RADs_all):0.4f}")
print(f"RAD>10%: {np.sum(RADs_all>0.1)/RADs_all.size:0.4f}")

# Total # weights flipped: 2250
# Max RAD: 0.7333
# RAD>10%: 0.0236

# boxplots: bit-flip index vs. RAD
plt.figure()
plt.boxplot([RADs_bf_idx[bf_idx] for bf_idx in range(32)])
plt.savefig("bf_idx-vs-RAD.jpg")
