from collections import defaultdict
from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x


def free_adv_train(
    model,
    data_tr,
    criterion,
    optimizer,
    lr_scheduler,
    eps,
    device,
    m=4,
    epochs=100,
    batch_size=128,
    dl_nw=10,
):
    """
    Free adversarial training, per Shafahi et al.'s work.
    Arguments:
    - model: randomly initialized model
    - data_tr: training dataset
    - criterion: loss function (e.g., nn.CrossEntropyLoss())
    - optimizer: optimizer to be used (e.g., SGD)
    - lr_scheduer: scheduler for updating the learning rate
    - eps: l_inf epsilon to defend against
    - device: device used for training
    - m: # times a batch is repeated
    - epochs: "virtual" number of epochs (equivalent to the number of
        epochs of standard training)
    - batch_size: training batch_size
    - dl_nw: number of workers in data loader
    Returns:
    - trained model
    """
    # init data loader
    loader_tr = DataLoader(
        data_tr, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=dl_nw
    )

    # init delta (adv. perturbation)
    delta = torch.zeros(data_tr[0][0].size()).to(device)

    # total number of updates
    # should be the same as in `standard_train()`
    total = epochs * len(data_tr) / batch_size

    # when to update lr
    scheduler_step_iters = int(np.ceil(len(data_tr) / batch_size))
    optimizer_steps = 0

    # train
    for epoch in tqdm(range(int(np.ceil(epochs / m)))):
        for batch_i, minibatch in enumerate(loader_tr, 0):
            # get inputs and labels
            inputs, labels = minibatch[0].to(device), minibatch[1].to(device)

            for m_i in range(m):
                # perturb
                noise = torch.autograd.Variable(
                    delta[: inputs.size(0)], requires_grad=True
                ).to(device)
                inputs = inputs + noise

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()

                # update perturbation
                delta[: inputs.size(0)] += eps * torch.sign(noise.grad)
                delta.clamp_(-eps, eps)

                # optimize
                optimizer.step()

                # update scheduler
                optimizer_steps += 1
                if optimizer_steps % scheduler_step_iters == 0:
                    lr_scheduler.step()

    # done
    return model


class SmoothedModel:
    """
    Use randomized smoothing to find L2 radius around sample x,
    s.t. the classification of x doesn't change within the L2 ball
    around x with probability >= 1-alpha.
    """

    ABSTAIN = -1

    def __init__(self, model, sigma):
        self.model = model
        self.sigma = sigma

    def _sample_under_noise(self, x, n, batch_size):
        """
        Classify input x under noise n times (with batch size
        equal to batch_size) and return class counts (i.e., an
        array counting how many times each class was assigned the
        max confidence).
        """

        with torch.no_grad():
            # class -> count, with a default of 0
            counts = defaultdict(lambda: 0)

            # generate `n` samples, in `batch_size` increments
            for _start in range(0, n, batch_size):
                # calculate size of current batch
                _end = min(_start + batch_size, n)
                size = _end - _start

                # create a batch of size `size` from single sample `x`
                batch = x.repeat(size, 1, 1, 1)

                # generate noise for each image in the batch
                noise = torch.normal(0, self.sigma, size=batch.size()).to(batch.device)

                # predict
                pred = self.model(batch + noise).argmax(1)

                # count
                for c in pred:
                    counts[c.item()] += 1

        return counts

    def certify(self, x, n0, n, alpha, batch_size):
        """
        Arguments:
        - model: pytorch classification model (preferably, trained with
            Gaussian noise)
        - sigma: Gaussian noise's sigma, for randomized smoothing
        - x: (single) input sample to certify
        - n0: number of samples to find prediction
        - n: number of samples for radius certification
        - alpha: confidence level
        - batch_size: batch size to use for inference
        Outputs:
        - prediction / top class (ABSTAIN in case of abstaining)
        - certified radius (0. in case of abstaining)
        """

        # find prediction (top class c)
        counts0 = self._sample_under_noise(x, n0, batch_size)
        c_a = list(counts0.keys())[np.argmax(counts0.values())]

        counts = self._sample_under_noise(x, n, batch_size)
        p_c = counts[c_a]

        # compute lower bound on p_c
        p_a = proportion_confint(p_c, n, alpha=2 * alpha, method="beta")[0]

        if p_a < 0.5:
            c, radius = self.ABSTAIN, 0.0
        else:
            c, radius = c_a, self.sigma * norm.ppf(p_a)

        # done
        return c, radius


class NeuralCleanse:
    """
    A method for detecting and reverse-engineering backdoors.
    """

    def __init__(
        self, model, dim=(1, 3, 32, 32), lambda_c=0.0005, step_size=0.005, niters=2000
    ):
        """
        Arguments:
        - model: model to test
        - dim: dimensionality of inputs, masks, and triggers
        - lambda_c: constant for balancing Neural Cleanse's objectives
            (l_class + lambda_c*mask_norm)
        - step_size: step size for SGD
        - niters: number of SGD iterations to find the mask and trigger
        """
        self.model = model
        self.dim = dim
        self.lambda_c = lambda_c
        self.niters = niters
        self.step_size = step_size
        self.loss_func = nn.CrossEntropyLoss()

    def find_candidate_backdoor(self, c_t, data_loader, device):
        """
        A method for finding a (potential) backdoor targeting class c_t.
        Arguments:
        - c_t: target class
        - data_loader: DataLoader for test data
        - device: device to run computation
        Outputs:
        - mask:
        - trigger:
        """

        # randomly initialize mask and trigger in [0,1]. mask is a single 2D image,
        # trigger is a 3D matrix, representing a 3-channel image.
        # TODO
        # mask_dim = list(self.dim)
        # mask_dim[1] = 1
        # mask_dim = tuple(mask_dim)
        # mask = torch.rand(mask_dim, requires_grad=True).to(device)
        mask = torch.rand(self.dim, requires_grad=True).to(device)
        trigger = torch.rand(self.dim, requires_grad=True).to(device)

        data_iter = cycle(data_loader)
        for _ in tqdm(range(self.niters)):
            inputs, _ = next(data_iter)
            inputs = inputs.to(device)
            target_labels = (
                torch.zeros(inputs.size(0), dtype=torch.long) + int(c_t)
            ).to(device)

            # build the triggered image according to the paper, A(x,m,t) = (1-m)*x + m*t
            inputs = (1 - mask) * inputs + mask * trigger

            # forward
            outputs = self.model(inputs)

            # calc loss against the target labels
            loss = self.loss_func(outputs, target_labels)
            # since we want to minimize the mask, we incorporate its norm to the loss
            loss += (self.lambda_c * mask.sum()).to(device)
            loss.backward()

            with torch.no_grad():
                mask -= mask.grad.sign() * self.step_size
                trigger -= trigger.grad.sign() * self.step_size

            # loss.backward()

            # optimize
            # optimizer.step()

            # sgd_iters += inputs.size(0)
            # if sgd_iters >= self.niters:
            #     break

        # done
        return mask, trigger
