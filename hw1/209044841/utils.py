import gzip
import struct
from os import path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import models


def load_pretrained_cnn(cnn_id, n_classes=4, models_dir="trained-models/"):
    """
    Loads one of the pre-trained CNNs that will be used throughout the HW
    """
    if not isinstance(cnn_id, int) or cnn_id < 0 or cnn_id > 2:
        raise ValueError(f"Unknown cnn_id {id}")
    model = eval(f"models.SimpleCNN{cnn_id}(n_classes=n_classes)")
    fpath = path.join(models_dir, f"simple-cnn-{cnn_id}")
    model.load_state_dict(torch.load(fpath))
    return model


class TMLDataset(Dataset):
    """
    Used to load the dataset used throughout the HW
    """

    def __init__(self, fpath="dataset.npz", transform=None):
        with gzip.open(fpath, "rb") as fin:
            self.data = np.load(fin, allow_pickle=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


def compute_accuracy(model, data_loader, device):
    """
    Evaluates and returns the (benign) accuracy of the model
    (a number in [0, 1]) on the labeled data returned by
    data_loader.
    """

    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in iter(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def run_whitebox_attack(attack, data_loader, targeted, device, n_classes=4):
    """
    Runs the white-box attack on the labeled data returned by
    data_loader. If targeted==True, runs targeted attacks, where
    targets are selected at random (t=c_x+randint(1, n_classes)%n_classes).
    Otherwise, runs untargeted attacks.
    The function returns:
    1- Adversarially perturbed sampels (one per input sample).
    2- True labels in case of untargeted attacks, and target labels in
       case of targeted attacks.
    """

    adv = []
    labels = []

    for x, y in iter(data_loader):
        x = x.to(device)
        y = y.to(device)

        if targeted:
            y = (y + torch.randint(1, n_classes, size=(len(y),)).to(device)) % n_classes

        x_adv = attack.execute(x, y, targeted)

        labels.append(y)
        adv.append(x_adv)

    return adv, labels


def run_blackbox_attack(attack, data_loader, targeted, device, n_classes=4):
    """
    Runs the black-box attack on the labeled data returned by
    data_loader. If targeted==True, runs targeted attacks, where
    targets are selected at random (t=(c_x+randint(1, n_classes))%n_classes).
    Otherwise, runs untargeted attacks.
    The function returns:
    1- Adversarially perturbed sampels (one per input sample).
    2- True labels in case of untargeted attacks, and target labels in
       case of targeted attacks.
    3- The number of queries made to create each adversarial example.
    """

    adv = []
    labels = []
    queries = []

    for x, y in iter(data_loader):
        x = x.to(device)
        y = y.to(device)

        if targeted:
            y = (y + torch.randint(1, n_classes, size=(len(y),)).to(device)) % n_classes

        x_adv, q = attack.execute(x, y, targeted)

        labels.append(y)
        adv.append(x_adv)
        queries.append(q)

    return adv, labels, torch.cat(queries)


def compute_attack_success(model, x_adv, y, batch_size, targeted, device):
    """
    Returns the success rate (a float in [0, 1]) of targeted/untargeted
    attacks. y contains the true labels in case of untargeted attacks,
    and the target labels in case of targeted attacks.
    """

    total = 0
    success = 0

    with torch.no_grad():
        for adv, labels in zip(x_adv, y):
            adv = adv.to(device)
            labels = labels.to(device)

            outputs = model(adv)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            if targeted:
                success += (predicted == labels).sum().item()
            else:
                success += (predicted != labels).sum().item()

    return success / total


def binary(num):
    """
    Given a float32, this function returns a string containing its
    binary representation (in big-endian, where the string only
    contains '0' and '1' characters).
    """
    # in big-endian
    return "".join([f"{c:08b}" for c in struct.pack(">f", num)])


def float32(binary):
    """
    This function inverts the "binary" function above. I.e., it converts
    binary representations of float32 numbers into float32 and returns the
    result.
    """
    # in big-endian
    return struct.unpack(">f", struct.pack(">I", int(binary, 2)))[0]


def random_bit_flip(w):
    """
    This function receives a weight in float32 format, picks a
    random bit to flip in it, flips the bit, and returns:
    1- The weight with the bit flipped
    2- The index of the flipped bit in {0, 1, ..., 31}
    """

    w = binary(w)
    idx = np.random.randint(0, len(w))
    w = w[:idx] + str(1 - int(w[idx])) + w[idx + 1 :]
    w = float32(w)

    return w, idx
